#include "magmadnn.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace magmadnn;
using namespace std;

const int NUM_NODES =  2709;
const int NUM_EDGES =  5430;

int main() {

    magmadnn_init();
    int cnt = 0;
    int cited, citing, paper_id;
    const int number_of_nodes = 2708, number_of_edges = 5429;
    const int TEST_SIZE = number_of_edges/10;
    const int TRAIN_SIZE = number_of_edges - TEST_SIZE;
    const int number_of_features = 1433;
    // The Cora dataset consists of 2708 scientific publications classified into one of seven classes. 
    // The citation network consists of 5429 links. 
    // Each publication in the dataset is described by a 0/1-valued word vector 
    // indicating the absence/presence of the corresponding word from the dictionary. 
    // The dictionary consists of 1433 unique words.

    // initialize the arrays for edges
    vector <int> u(NUM_EDGES-1), v(NUM_EDGES-1), edge_id(NUM_EDGES-1);
    int neg_edge_id[5430];
    vector <int> test_pos_u(TEST_SIZE), test_pos_v(TEST_SIZE), test_neg_u(TEST_SIZE), test_neg_v(TEST_SIZE); 
    vector <int> train_pos_u(TRAIN_SIZE), train_pos_v(TRAIN_SIZE), train_neg_u(TRAIN_SIZE), train_neg_v(TRAIN_SIZE);
    vector <int> node_id(number_of_nodes);

    auto X = op::var<float> ("X", {number_of_nodes,number_of_features}, {ZERO, {}}, HOST);

    ifstream coracites("./data/cora/cora.cites");
    if (coracites.is_open()) cout<<"cora cites is open"<<endl;

    // check and read edges from file
    while (cnt<number_of_edges && coracites >> cited && coracites >> citing) {
        u[cnt] = cited;
        v[cnt] = citing;
        edge_id[cnt] = cnt;
        cnt ++;
    }   

    // convert the paper ids to zero based indices
    // the .content file contains descriptions of the papers in the following format: <paper_id> <word_attributes>+ <class_label>
    // The first entry in each line contains the unique string ID of the paper followed by binary values indicating whether each word in the vocabulary is present (indicated by 1) or absent (indicated by 0) in the paper. 
    // Finally, the last entry in the line contains the class label of the paper.

    ifstream coracontent("./data/cora/cora.content");
    if (coracontent.is_open()) cout<<"cora content is open" <<endl;
    cnt = 0;
    float temp_float;
    string temp_string;
    while (cnt < number_of_nodes && coracontent >> paper_id) {
        node_id[cnt] = paper_id;
        for (int i = 0; i<1433;i++) {
            coracontent>>temp_float;
            X -> get_output_tensor() -> set({cnt, i}, (float) temp_float);
        }
        coracontent>> temp_string;
        cnt++;
    }

    for (int i=0;i<number_of_edges;i++) {
        for (int j=0;j<number_of_nodes;j++) {
            if (u[i] == node_id[j]) {
                u[i] = j;
            }
            if (v[i] ==  node_id[j]) {
                v[i] = j;
            }
        }
    }


    random_shuffle(edge_id.begin(), edge_id.end());

    for (int i=0;i<TEST_SIZE;i++) {
        test_pos_u[i] = u[edge_id[i]];
        test_pos_v[i] = v[edge_id[i]];
    }
    for (int i=TEST_SIZE; i<number_of_edges;i++) {
        train_pos_u[i-TEST_SIZE] = u[edge_id[i]];
        train_pos_v[i-TEST_SIZE] = v[edge_id[i]];
    }



    // find all the negative edges and split them for training and testing---------------------------------------------------------------

    //adj matrix (A_hat)
    auto adj = op::var<float> ("adj", {number_of_nodes,number_of_nodes}, {ZERO, {}}, HOST);
    for (int i=0;i<number_of_edges;i++) {
        adj -> get_output_tensor() -> set({u[i], v[i]}, (float)1.f);
        adj -> get_output_tensor() -> set({v[i], u[i]}, (float)1.f);
    }
    for (int i=0;i<number_of_nodes;i++) {
        adj -> get_output_tensor() -> set({i,i}, (float)1.f);
    }

    // we only sample a number of non existent edges as negative examples
    // there are 5429 positive edges so we can sample 5429 negative edges
    vector <int> neg_u(number_of_edges), neg_v(number_of_edges);

    Tensor <float> *adj_tensor = adj -> eval();

    Tensor<float> flags (adj_tensor -> get_shape(), {NONE, {}}, HOST);
    flags.copy_from(*adj_tensor);

    int edge_cnt = 0;
    int rand_u, rand_v;

    while (edge_cnt<number_of_edges) {
        rand_u = rand() % number_of_nodes;
        rand_v = rand() % number_of_nodes;
        if (flags.get({rand_u, rand_v}) == (float)0.f) {
            flags.set({rand_u, rand_v}, (float) 1.f);
            neg_u[edge_cnt] = rand_u;
            neg_v[edge_cnt] = rand_v;
            edge_cnt ++;
        }
    }
    // split
    for (int i=0;i<TEST_SIZE;i++) {
        test_neg_u[i] = neg_u[i];
        test_neg_v[i] = neg_v[i];
    }
    for (int i=TEST_SIZE; i<number_of_edges;i++) {
        train_neg_u[i-TEST_SIZE] = neg_u[i];
        train_neg_v[i-TEST_SIZE] = neg_v[i];
    }

    // remove testing edges from the graph
    Tensor<float> a_hat (adj_tensor -> get_shape(), {NONE, {}}, HOST);
    a_hat.copy_from(*adj_tensor);
    for (int i=0;i<TEST_SIZE;i++) {
        a_hat.set({test_pos_u[i], test_pos_v[i]}, (float) 0.f);
    }
    auto A_hat = op::var <float> ("A hat", &a_hat);


    // calculate the inverse of the degree matrix D_hat--------------------------------------------------------------
    // calculate degree vector for all nodes
    auto ones = op::var <float> ("ones", {1, number_of_nodes}, {ONE, {}}, HOST);
    auto D = op::matmul(ones, A_hat);
    Tensor <float> *D_tensor = D -> eval();

    // calculate the reciprocal of the degree (diagonal matrix)
    auto D_rec = op::var <float> ("D_rec", {number_of_nodes, number_of_nodes}, {ZERO, {}}, HOST);
    for (int i=0; i<number_of_nodes; i++) 
    {
        float degree = D -> get_output_tensor() -> get({0, i});
        D_rec -> get_output_tensor() -> set({i,i}, (float)1.f/degree);
    }
    Tensor <float> *Drec_tensor = D_rec -> eval();

    // testing
    int num_pos_edges = train_pos_u.size();
    int num_neg_edges = train_neg_u.size();
    int num_edges = num_pos_edges + num_neg_edges;
    Tensor<float> testing_output ({1, num_edges}, {ZERO, {}}, HOST);
    int a, b;
    float sum, feat1, feat2;
    for (int i=0;i<num_pos_edges;i++) {
            sum = 0;
            a = train_pos_u[i];
            b = train_pos_v[i];
            for (int j=0;j<number_of_features;j++) {
                feat1 = X -> get_output_tensor() -> get({a, j});
                feat2 = X -> get_output_tensor() -> get({b, j});
                sum += feat1 * feat2;
            }
            testing_output.set({0, i}, sum);
    }
    io::print_tensor(&testing_output);


    // the model ----------------------------------------------------------------------------------------------------


    layer::InputLayer<float> *input = layer::input(X);
    layer::GCNLayer<float> *conv1 = layer::gcn_layer(A_hat, D_rec, input->out());
    layer::ActivationLayer<float> *relu1 = layer::activation(conv1->out(), layer::RELU);
    layer::GCNLayer<float> *conv2 = layer::gcn_layer(A_hat, D_rec, relu1->out());
    layer::OutputLayer<float> *output = layer::output(conv2->out());
    
    // need to build a decoder layer
    // input train pos and train neg edges and X
    // output: pos score + neg score combined
    
    // close the file
    coracites.close();
    coracontent.close();
    
    delete adj;
    delete X;
    delete A_hat;
    magmadnn_finalize();
    return 0;
}