#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include "hplelib.h"

char data[MAX_STRING], task[MAX_STRING];
char file_path[MAX_STRING], output_path[MAX_STRING], mode;
int binary = 0, num_threads = 1, vector_size = 100, negative = 5, iters = 10, iter;
long long node_count_actual;
std::vector<int> mention_set, feature_set, type_set;
real starting_lr = 0.025, alpha = 0.0001;
real lr;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;
ransampl_ws* ws;

line_node node_M, node_F, node_Y;
line_link link_MF, link_MY, link_FY, link_YY;

void *train_BCD_thread(void *id)
{
    unsigned long long next_random = (long long)id;
    int node_size = node_M.get_num_nodes(), T = 0;    
    long long tid = (long long)id;
    int begin = node_size / num_threads * tid;
    int end = node_size / num_threads * (tid + 1);
    if (tid == num_threads) end = node_size;

    for (int k = begin; k != end; k++, T++)
    {
        if (T % 10000 == 0)
        {
            node_count_actual += 10000;
            printf("%cLR: %f Epoch: %d/%d Progress: %.3lf%%", 13, lr, iter + 1, iters, node_count_actual / (real)(node_size + 1) / iters * 100);
            fflush(stdout);
        }
        int u = mention_set[k];
        link_MF.train_BCD_sg(lr, next_random, u);
        // link_MY.train_BCD_sg(lr, next_random, u);
        link_MY.train_BCD_ple(lr, alpha, next_random, u);
    }
    pthread_exit(NULL);
}

void *train_BCD_thread_F(void *id)
{
    unsigned long long next_random = (long long)id;    
    int node_size = node_F.get_num_nodes(), T = 0;
    long long tid = (long long)id;
    int begin = node_size / num_threads * tid;
    int end = node_size / num_threads * (tid + 1);
    if (tid == num_threads) end = node_size;

    for (int k = begin; k != end; k++, T++)
    {
        int u = feature_set[k];
        link_FY.train_BCD_sg(lr, next_random, u);
    }
    pthread_exit(NULL);
}

void *train_BCD_thread_Y(void *id)
{
    unsigned long long next_random = (long long)id;
    
    int node_size = node_Y.get_num_nodes(), T = 0;    
    long long tid = (long long)id;
    int begin = node_size / num_threads * tid;
    int end = node_size / num_threads * (tid + 1);
    if (tid == num_threads) end = node_size;

    for (int k = begin; k != end; k++, T++)
    {
        int u = type_set[k];
        link_YY.train_BCD_sg(lr, next_random, u);
    }
    pthread_exit(NULL);
}

void TrainModel() {
    long a;
    pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    char file_name[MAX_STRING];
    printf("Mode: %c\n", mode);
    
    sprintf(file_name, "%smention.txt", file_path);
    node_M.init(file_name, vector_size);
    sprintf(file_name, "%sfeature.txt", file_path);
    node_F.init(file_name, vector_size);
    sprintf(file_name, "%stype.txt", file_path);
    node_Y.init(file_name, vector_size);
    
    sprintf(file_name, "%smention_feature.txt", file_path);
    link_MF.init(file_name, &node_M, &node_F, negative); // graph is inversed
    // link_FM.init_transpose(file_name, &node_F, &node_M, negative); // graph is inversed
    
    sprintf(file_name, "%smention_type.txt", file_path);
    link_MY.init(file_name, &node_M, &node_Y, negative);
    
    sprintf(file_name, "%sfeature_type.txt", file_path);
    link_FY.init(file_name, &node_F, &node_Y, negative);

    sprintf(file_name, "%stype_type_kb.txt", file_path);
    link_YY.init(file_name, &node_Y, &node_Y, negative);

    for (int k = 0; k != node_M.get_num_nodes(); k++) mention_set.push_back(k);
    for (int k = 0; k != node_F.get_num_nodes(); k++) feature_set.push_back(k);
    for (int k = 0; k != node_Y.get_num_nodes(); k++) type_set.push_back(k);

    gsl_rng_env_setup();
    gsl_T = gsl_rng_rand48;
    gsl_r = gsl_rng_alloc(gsl_T);
    gsl_rng_set(gsl_r, 314159265);
    
    clock_t start = clock();
    printf("Training process:\n");
    if (mode == 'b')
    {
        for (iter = 0; iter != iters; iter++)
        {
            //// Varying learning rate
            // lr = starting_lr * (1 - iter / (real)(iters));

            //// update MF + MY network
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, train_BCD_thread, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

            //// update FY, YY networks
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, train_BCD_thread_F, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
            for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, train_BCD_thread_Y, (void *)a);
            for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);

            //// SG model node updating
            node_F.update_vec();
            //// PLE model node updating
            node_M.update_vec_ple(lr, alpha);
            node_Y.update_vec_ple(lr, alpha);
        }
    }
    printf("\n");
    clock_t finish = clock();
    printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);
    
    sprintf(file_name, "%semb_hple_corrKB_mention.txt", output_path);
    node_M.output(file_name, binary);
    sprintf(file_name, "%semb_hple_corrKB_feature.txt", output_path);
    node_F.output(file_name, binary);
    sprintf(file_name, "%semb_hple_corrKB_type.txt", output_path);
    node_Y.output(file_name, binary);
}


int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
        if (a == argc - 1) {
            printf("Argument missing for %s\n", str);
            exit(1);
        }
        return a;
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("HPLE\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-data <path>\n");
        printf("\t\tData (FIGER / BBN)\n");
        printf("\t-task <path>\n");
        printf("\t\tTask (reduce_label_noise / typing)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of embedding; default is 100\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 5 - 10 (0 = not used)\n");
        printf("\t-iters <int>\n");
        printf("\t\tSet the number of iterations as <int>\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 1)\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the value of weight decay (default 0.0001)\n");
        printf("\t-lr <float>\n");
        printf("\t\tSet the value of learning rate (default 0.025)\n");
        return 0;
    }
    if ((i = ArgPos((char *)"-data", argc, argv)) > 0) strcpy(data, argv[i + 1]);
    if ((i = ArgPos((char *)"-task", argc, argv)) > 0) strcpy(task, argv[i + 1]);
    if ((i = ArgPos((char *)"-mode", argc, argv)) > 0) mode = argv[i + 1][0];
    if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-iters", argc, argv)) > 0) iters = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-lr", argc, argv)) > 0) starting_lr = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
    sprintf(file_path, "data/%s/%s/no/", data, task);
    sprintf(output_path, "data/%s/%s/no/Results/", data, task);
    lr = starting_lr;
    TrainModel();
    return 0;
}