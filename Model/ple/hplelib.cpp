#include "hplelib.h"

real sigmoid(real x)
{
    return 1.0 / (1.0 + exp(-x));
}

line_node::line_node() : vec(NULL, 0, 0), err(NULL, 0, 0)
{
    node = NULL;
    node_size = 0;
    vector_size = 0;
    node_file[0] = 0;
    _vec = NULL;
    _err = NULL;
}

line_node::~line_node()
{
    if (node != NULL) {free(node); node = NULL;}
    node_size = 0;
    vector_size = 0;
    node_file[0] = 0;
    if (_vec != NULL) {free(_vec); _vec = NULL;}
    new (&vec) Eigen::Map<BLPMatrix>(NULL, 0, 0);
    if (_err != NULL) {free(_err); _err = NULL;}
    new (&_err) Eigen::Map<BLPMatrix>(NULL, 0, 0);
}

int line_node::get_vector_dim()
{
    return vector_size;
}

int line_node::get_num_nodes()
{
    return node_size;
}

real *line_node::get_vector()
{
    return _vec;
}

struct struct_node *line_node::get_node()
{
    return node;
}

void line_node::init(char *file_name, int vector_dim)
{
    strcpy(node_file, file_name);
    vector_size = vector_dim;
    
    char str[2 * MAX_STRING + 10000];
    int pst;
    
    // read the node file to get the number of nodes
    FILE *fi = fopen(node_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: node file not found!\n");
        printf("%s\n", node_file);
        exit(1);
    }
    node_size = 0;
    while (fgets(str, sizeof(str), fi))
    {
        int len = strlen(str);
        int k = 0;
        for (k = 0; k != len; k++) if (str[k] == '\t')
            break;
        pst = atoi(str + k + 1);
        if (pst >= node_size) node_size = pst + 1;
    }
    fclose(fi);
    
    node = (struct struct_node *)calloc(node_size, sizeof(struct struct_node));
    
    // read node name and node id
    fi = fopen(node_file, "rb");
    while (fgets(str, sizeof(str), fi))
    {
        int len = strlen(str);
        int k = 0;
        for (k = 0; k != len; k++) if (str[k] == '\t')
            break;
        str[k] = 0;
        pst = atoi(str + k + 1);
        
        unsigned int length = strlen(str) + 1;
        if (length > MAX_STRING) length = MAX_STRING;
        str[MAX_STRING - 1] = 0;
        //node[pst].word = (char *)malloc(length * sizeof(char));
        strcpy(node[pst].word, str);
    }
    fclose(fi);
    
    // initialize the embedding
    long long a, b;
    a = posix_memalign((void **)&_vec, 128, (long long)node_size * vector_size * sizeof(real));
    if (_vec == NULL) { printf("Memory allocation failed\n"); exit(1); }
    for (b = 0; b < vector_size; b++) for (a = 0; a < node_size; a++)
        _vec[a * vector_size + b] = (rand() / (real)RAND_MAX - 0.5) / vector_size;
    new (&vec) Eigen::Map<BLPMatrix>(_vec, node_size, vector_size);
    
    a = posix_memalign((void **)&_err, 128, (long long)node_size * vector_size * sizeof(real));
    if (_err == NULL) { printf("Memory allocation failed\n"); exit(1); }
    for (b = 0; b < vector_size; b++) for (a = 0; a < node_size; a++)
        _err[a * vector_size + b] = 0;
    new (&err) Eigen::Map<BLPMatrix>(_err, node_size, vector_size);
    
    // initialize the cnt of gradients for a node's err vector
    grad_cnt = (int *)malloc(node_size * sizeof(int));
    for (int k = 0; k != node_size; k++) grad_cnt[k] = 1; // initialize as 1
    
    printf("Reading nodes from file: %s, DONE!\n", node_file);
    printf("Node size: %d\n", node_size);
    printf("Node dims: %d\n", vector_size);
}


void line_node::output(char *file_name, int binary)
{
    FILE *fo = fopen(file_name, "wb");
    if (binary == 1)
    {
        fprintf(fo, "%d %d\n", node_size, vector_size);
        for (int a = 0; a != node_size; a++)
        {
            fprintf(fo, "%s ", node[a].word);
            for (int b = 0; b != vector_size; b++) fwrite(&_vec[a * vector_size + b], sizeof(real), 1, fo);
            fprintf(fo, "\n");
        }
    }
    if (binary == 0)
    {
        fprintf(fo, "%d %d\n", node_size, vector_size);
        for (int k = 0; k != node_size; k++)
        {
            fprintf(fo, "%s\t%d\t", node[k].word, k);
            for (int c = 0; c != vector_size; c++) fprintf(fo, "%lf ", _vec[k * vector_size + c]);
            fprintf(fo, "\n");
        }
    }
    fclose(fo);
}

void line_node::update_vec_ple(real lr, real alpha)
{
    // term from Partial-label objective
    for (int k = 0; k != node_size; k++) vec.row(k) += err.row(k) / grad_cnt[k] - lr * alpha * vec.row(k);
    err.setZero();
    for (int k = 0; k != node_size; k++) grad_cnt[k] = 1;
}

void line_node::update_vec()
{
    for (int k = 0; k != node_size; k++) vec.row(k) += err.row(k) / grad_cnt[k];
    err.setZero();
    for (int k = 0; k != node_size; k++) grad_cnt[k] = 1;
}

line_link::line_link()
{
    node_u = NULL;
    node_v = NULL;
    expTable = NULL;
    ws = NULL;
    edge_cnt = 0;
    edge_u = NULL;
    edge_v = NULL;
    edge_w = NULL;
    link_file[0] = 0;
    graph = NULL;
}

line_link::~line_link()
{
    node_u = NULL;
    node_v = NULL;
    if (expTable != NULL) {free(expTable); expTable = NULL;}
    if (ws != NULL) {ransampl_free(ws); ws = NULL;}
    edge_cnt = 0;
    if (edge_u != NULL) {free(edge_u); edge_u = NULL;}
    if (edge_v != NULL) {free(edge_v); edge_v = NULL;}
    if (edge_w != NULL) {free(edge_w); edge_w = NULL;}
    link_file[0] = 0;
    if (graph != NULL) {delete [] graph; graph = NULL;}
    
}

void line_link::init(char *file_name, line_node *p_u, line_node *p_v, int negative)
{
    strcpy(link_file, file_name);
    node_u = p_u;
    node_v = p_v;
    neg_samples = negative;

    if (node_u->get_vector_dim() != node_v->get_vector_dim())
    {
        printf("ERROR: vector dimsions are not same!\n");
        exit(1);
    }
    
    dgr_u = (double *)calloc(node_u->node_size, sizeof(double));
    dgr_v = (double *)calloc(node_v->node_size, sizeof(double));
    
    // compute the number of edges
    char str[2 * MAX_STRING + 10000];
    FILE *fi = fopen(link_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: link file not found!\n");
        printf("%s\n", link_file);
        exit(1);
    }
    edge_cnt = 0;
    while (fgets(str, sizeof(str), fi)) edge_cnt++;
    fclose(fi);
    
    // allocate spaces
    int u, v;
    double wei;
    edge_u = (int *)malloc(edge_cnt * sizeof(int));
    edge_v = (int *)malloc(edge_cnt * sizeof(int));
    edge_w = (double *)malloc(edge_cnt * sizeof(double));
    if (edge_u == NULL || edge_v == NULL || edge_w == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    graph = new std::vector<struct_neighbor>[node_u->node_size];
    nb_set = new std::set<int>[node_u->node_size];
    struct_neighbor cur_nb;
    
    // read edges
    fi = fopen(link_file, "rb");
    for (int k = 0; k != edge_cnt; k++)
    {
        fscanf(fi, "%d %d %lf", &u, &v, &wei);
        
        if (k % 10000 == 0)
        {
            printf("Reading edges: %.3lf%%%c", k / (double)(edge_cnt + 1) * 100, 13);
            fflush(stdout);
        }
        
        // store edges
        edge_u[k] = u;
        edge_v[k] = v;
        edge_w[k] = wei;
        
        // update degrees
        dgr_u[u] += wei;
        dgr_v[v] += wei;
        
        // update graph
        cur_nb.index = v;
        cur_nb.wei = wei;
        graph[u].push_back(cur_nb);
        
        // update neibor set
        nb_set[u].insert(v);
    }
    fclose(fi);
    
    // initialize edge sampler
    ws = ransampl_alloc(edge_cnt);
    ransampl_set(ws, edge_w);
    
    // compute the value of exp function before training
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    
    // initialize the negative sampling table
    int a, i;
    double total_pow = 0, d1;
    real power = 0.75;
    
    neg_table_u = (int *)calloc(neg_table_size, sizeof(int));
    neg_table_v = (int *)calloc(neg_table_size, sizeof(int));
    
    total_pow = 0;
    for (a = 0; a < node_u->node_size; a++) total_pow += pow(dgr_u[a], power);
    i = 0;
    d1 = pow(dgr_u[i], power) / (real)total_pow;
    for (a = 0; a < neg_table_size; a++) {
        neg_table_u[a] = i;
        if (a / (real)neg_table_size > d1) {
            i++;
            d1 += pow(dgr_u[i], power) / (real)total_pow;
        }
        if (i >= node_u->node_size) i = node_u->node_size - 1;
    }
    
    total_pow = 0;
    for (a = 0; a < node_v->node_size; a++) total_pow += pow(dgr_v[a], power);
    i = 0;
    d1 = pow(dgr_v[i], power) / (real)total_pow;
    for (a = 0; a < neg_table_size; a++) {
        neg_table_v[a] = i;
        if (a / (real)neg_table_size > d1) {
            i++;
            d1 += pow(dgr_v[i], power) / (real)total_pow;
        }
        if (i >= node_v->node_size) i = node_v->node_size - 1;
    }

    printf("Reading edges from file: %s, DONE!\n", link_file);
    printf("Edge size: %lld\n", edge_cnt);
}

void line_link::init_transpose(char *file_name, line_node *p_u, line_node *p_v, int negative)
{
    strcpy(link_file, file_name);
    node_u = p_u;
    node_v = p_v;
    neg_samples = negative;

    if (node_u->get_vector_dim() != node_v->get_vector_dim())
    {
        printf("ERROR: vector dimsions are not same!\n");
        exit(1);
    }
    
    dgr_u = (double *)calloc(node_u->node_size, sizeof(double));
    dgr_v = (double *)calloc(node_v->node_size, sizeof(double));
    
    // compute the number of edges
    char str[2 * MAX_STRING + 10000];
    FILE *fi = fopen(link_file, "rb");
    if (fi == NULL)
    {
        printf("ERROR: link file not found!\n");
        printf("%s\n", link_file);
        exit(1);
    }
    edge_cnt = 0;
    while (fgets(str, sizeof(str), fi)) edge_cnt++;
    fclose(fi);
    
    // allocate spaces
    int u, v;
    double wei;
    edge_u = (int *)malloc(edge_cnt * sizeof(int));
    edge_v = (int *)malloc(edge_cnt * sizeof(int));
    edge_w = (double *)malloc(edge_cnt * sizeof(double));
    if (edge_u == NULL || edge_v == NULL || edge_w == NULL)
    {
        printf("Error: memory allocation failed!\n");
        exit(1);
    }
    graph = new std::vector<struct_neighbor>[node_u->node_size];
    nb_set = new std::set<int>[node_u->node_size];
    struct_neighbor cur_nb;
    
    // read edges
    fi = fopen(link_file, "rb");
    for (int k = 0; k != edge_cnt; k++)
    {
        fscanf(fi, "%d %d %lf", &v, &u, &wei);
        
        if (k % 10000 == 0)
        {
            printf("Reading edges: %.3lf%%%c", k / (double)(edge_cnt + 1) * 100, 13);
            fflush(stdout);
        }
        
        // store edges
        edge_u[k] = u;
        edge_v[k] = v;
        edge_w[k] = wei;
        
        // update degrees
        dgr_u[u] += wei;
        dgr_v[v] += wei;
        
        // update graph
        cur_nb.index = v;
        cur_nb.wei = wei;
        graph[u].push_back(cur_nb);
        
        // update neibor set
        nb_set[u].insert(v);
    }
    fclose(fi);
    
    // initialize edge sampler
    ws = ransampl_alloc(edge_cnt);
    ransampl_set(ws, edge_w);
    
    // compute the value of exp function before training
    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }
    
    // initialize the negative sampling table
    int a, i;
    double total_pow = 0, d1;
    real power = 0.75;
    
    neg_table_u = (int *)calloc(neg_table_size, sizeof(int));
    neg_table_v = (int *)calloc(neg_table_size, sizeof(int));
    
    total_pow = 0;
    for (a = 0; a < node_u->node_size; a++) total_pow += pow(dgr_u[a], power);
    i = 0;
    d1 = pow(dgr_u[i], power) / (real)total_pow;
    for (a = 0; a < neg_table_size; a++) {
        neg_table_u[a] = i;
        if (a / (real)neg_table_size > d1) {
            i++;
            d1 += pow(dgr_u[i], power) / (real)total_pow;
        }
        if (i >= node_u->node_size) i = node_u->node_size - 1;
    }
    
    total_pow = 0;
    for (a = 0; a < node_v->node_size; a++) total_pow += pow(dgr_v[a], power);
    i = 0;
    d1 = pow(dgr_v[i], power) / (real)total_pow;
    for (a = 0; a < neg_table_size; a++) {
        neg_table_v[a] = i;
        if (a / (real)neg_table_size > d1) {
            i++;
            d1 += pow(dgr_v[i], power) / (real)total_pow;
        }
        if (i >= node_v->node_size) i = node_v->node_size - 1;
    }

    printf("Reading edges from file: %s, DONE!\n", link_file);
    printf("Edge size: %lld\n", edge_cnt);
}


void line_link::train_miniBatch_sg(real *_error_vec_u, real *_error_vec_v, real lr, unsigned long long &rand_index, int u)
{
    int target, v, vector_size;
    // int label;
    double w;
    real f, g;
    
    vector_size = node_u->vector_size;
    Eigen::Map<BLPVector> error_vec_u(_error_vec_u, vector_size);
    Eigen::Map<BLPVector> error_vec_v(_error_vec_v, vector_size);
    
    // for each neighbor node v of node u
    for (unsigned int k = 0; k != graph[u].size(); k++)
    {
        v = graph[u][k].index;
        w = graph[u][k].wei;
        // w = 1.0; // set to unweighted edge

        // initialize the error vector
        error_vec_u.setZero();
        error_vec_v.setZero();
        
        // for positive example (u, v)
        f = node_u->vec.row(u) * node_v->vec.row(v).transpose(); // dot product
        f = sigmoid(f);
        g = (1 - f) * lr * w;
        
        error_vec_u += g * node_v->vec.row(v);
        error_vec_v += g * node_u->vec.row(u);
        
        // for negative example u->v'
        for (int d = 0; d < neg_samples; d++)
        {
            // sample a v' from the negative sampling table
            rand_index = rand_index * (unsigned long long)25214903917 + 11;
            target = neg_table_v[(rand_index >> 16) % neg_table_size];
            if (target == v) continue;
            
            f = node_u->vec.row(u) * node_v->vec.row(target).transpose();
            f = sigmoid(f);
            g = - f * lr * w;

            error_vec_u += g * node_v->vec.row(target);
            node_v->vec.row(target) += g * node_u->vec.row(u);
        }
        
        // for negative example u'->v
        for (int d = 0; d < neg_samples; d++)
        {
            // sample a u' from the negative sampling table
            rand_index = rand_index * (unsigned long long)25214903917 + 11;
            target = neg_table_u[(rand_index >> 16) % neg_table_size];
            if (target == u) continue;
            
            f = node_u->vec.row(target) * node_v->vec.row(v).transpose();
            f = sigmoid(f);
            g = - f * lr * w;

            error_vec_v += g * node_u->vec.row(target);
            node_u->vec.row(target) += g * node_v->vec.row(v);
        }
        
        node_u->vec.row(u) += error_vec_u;
        node_v->vec.row(v) += error_vec_v;
    }
}

void line_link::train_miniBatch_ple(real *_error_vec_u, real lr, real alpha, unsigned long long &rand_index, int u)
{
    int pos, neg, v, vector_size;
    double max_value_pos, max_value_neg;
    // double w;
    real f, margin = 1;
    
    vector_size = node_u->vector_size;
    Eigen::Map<BLPVector> error_vec_u(_error_vec_u, vector_size);
    
    // initialize the error vector
    error_vec_u.setZero();
    
    pos = -1;
    max_value_pos = -1000000000;
    for (unsigned int k = 0; k != graph[u].size(); k++)
    {
        v = graph[u][k].index;
        // w = graph[u][k].wei;
        
        f = node_u->vec.row(u) * node_v->vec.row(v).transpose();
        if (f > max_value_pos)
        {
            max_value_pos = f;
            pos = v;
        }
    }
    
    neg = -1;
    max_value_neg = -1000000000;
    for (v = 0; v != node_v->node_size; v++)
    {
        if (nb_set[u].count(v) == 1) continue; // skip neighbor node v of node u
        
        f = node_u->vec.row(u) * node_v->vec.row(v).transpose();
        if (f > max_value_neg)
        {
            max_value_neg = f;
            neg = v;
        }
    }

    if (pos == -1 || neg == -1) return;
    
    // update {u, v_pos, v_neg} embeddings
    error_vec_u = -alpha * node_u->vec.row(u);
    if (max_value_pos - max_value_neg < margin)
    {
        error_vec_u -= node_v->vec.row(neg) - node_v->vec.row(pos);
        node_v->vec.row(pos) += lr * node_u->vec.row(u);
        node_v->vec.row(neg) -= lr * node_u->vec.row(u);
    }
    node_u->vec.row(u) += lr * error_vec_u;
}

// Skip-gram by Block Coordinate Descent
void line_link::train_BCD_sg(real lr, unsigned long long &rand_index, int u)
{
    int target, v;
    double w;
    real f, g;
    
    for (unsigned int k = 0; k != graph[u].size(); k++)
    {
        // positive example (u, v)
        v = graph[u][k].index;
        w = graph[u][k].wei;
        
        f = node_u->vec.row(u) * node_v->vec.row(v).transpose(); // dot product
        f = sigmoid(f);
        g = (1 - f) * lr * w;
        
        node_u->err.row(u) += g * node_v->vec.row(v);
        node_v->err.row(v) += g * node_u->vec.row(u);
        // node_u->grad_cnt[u]++;
        // node_v->grad_cnt[v]++;


        // negative examples {v'} of u
        for (int d = 0; d < neg_samples; d++)
        {
            // sample a v' from the negative sampling table
            rand_index = rand_index * (unsigned long long)25214903917 + 11;
            target = neg_table_v[(rand_index >> 16) % neg_table_size];
            if (target == v) continue;
            
            f = node_u->vec.row(u) * node_v->vec.row(target).transpose();
            f = sigmoid(f);
            g = - f * lr * w;
            node_u->err.row(u) += g * node_v->vec.row(target);
            node_v->err.row(target) += g * node_u->vec.row(u);
            
            node_u->grad_cnt[u]++;
            node_v->grad_cnt[target]++; 
        }
        
        // negative examples {u'} of v
        for (int d = 0; d < neg_samples; d++)
        {
            // sample a u' from the negative sampling table
            rand_index = rand_index * (unsigned long long)25214903917 + 11;
            target = neg_table_u[(rand_index >> 16) % neg_table_size];
            if (target == u) continue;
            
            f = node_u->vec.row(target) * node_v->vec.row(v).transpose();
            f = sigmoid(f);
            g = - f * lr * w;
            node_v->err.row(v) += g * node_u->vec.row(target);
            node_u->err.row(target) += g * node_v->vec.row(v);
            
            node_u->grad_cnt[target]++;
            node_v->grad_cnt[v]++;
        }
    }
}

// Partial-Label Embedding by Block Coordinate Descent
void line_link::train_BCD_ple(real lr, real alpha, unsigned long long &rand_index, int u)
{
    int v, v_pos, v_neg;
    double max_value_pos, max_value_neg;
    real f, margin = 1.0;
    
    v_pos = -1;
    max_value_pos = -1000000000;
    for (unsigned int k = 0; k != graph[u].size(); k++)
    {
        v = graph[u][k].index;        
        f = node_u->vec.row(u) * node_v->vec.row(v).transpose();
        if (f > max_value_pos)
        {
            max_value_pos = f;
            v_pos = v;
        }
    }
    
    v_neg = -1;
    max_value_neg = -1000000000;
    for (v = 0; v != node_v->node_size; v++)
    {
        if (nb_set[u].count(v) == 1) continue;
        
        f = node_u->vec.row(u) * node_v->vec.row(v).transpose();
        if (f > max_value_neg)
        {
            max_value_neg = f;
            v_neg = v;
        }
    }

    if (v_pos == -1 || v_neg == -1) return;
    
    // node_u->err.row(u) += -lr * alpha * node_u->vec.row(u);
    // node_v->err.row(v_pos) += -lr * alpha * node_v->vec.row(v_pos);
    // node_v->err.row(v_neg) += -lr * alpha * node_v->vec.row(v_neg);

    if (max_value_pos - max_value_neg < margin)
    {
        node_u->err.row(u) += lr * (node_v->vec.row(v_pos) - node_v->vec.row(v_neg));
        node_v->err.row(v_pos) += lr * node_u->vec.row(u);
        node_v->err.row(v_neg) += -lr * node_u->vec.row(u);
    }

    //// old
    node_u->grad_cnt[u]++;
    node_v->grad_cnt[v_pos]++;
    node_v->grad_cnt[v_neg]++;
}
