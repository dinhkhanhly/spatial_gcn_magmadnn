/**
* @file gcn_decoder.cpp 
*/

#include"layer/gcn_decoder/gcn_decoder.h"
#include "tensor/tensor_io.h"

using namespace std;

namespace magmadnn {
namespace layer {
        
    template<typename T>
	GCNDecoder<T>::GCNDecoder(vector <int> train_pos_u, vector <int> train_pos_v, 
                    vector <int> train_neg_u, vector <int> train_neg_v, op::Operation<T> *X)
	:Layer<T>::Layer(X->get_output_shape(), X),train_pos_u(train_pos_u), train_pos_v(train_pos_v), 
                train_neg_u(train_neg_u), train_neg_v(train_neg_v) {		
		init();
	}
    template<typename T>
	GCNDecoder<T>::~GCNDecoder() {
		delete this->weight;
	}
	
	template<typename T>
	std::vector<op::Operation<T> *> GCNDecoder<T>::get_weights() {
		return {this->weight};
	}

    template<typename T>
	void GCNDecoder<T>::init() {
		this->name = "GCNLayer";

		this->wTensor = new Tensor<T>({this->X->get_output_shape()[1],this->X->get_output_shape()[1]},
				{UNIFORM, {(T)0,(T)1}}, HOST);
		
		this->weight = op::var("__" + this->name + "_layer_weight", wTensor);
		cout<<"weight"<<endl;
		std::cout<<this->weight->get_output_shape(0) << " " << this->weight-> get_output_shape(1)<<endl;
		std::cout<<"X"<<endl;
		std::cout<<this->X->get_output_shape(0) << " " << this->X-> get_output_shape(1)<<endl;
		std::cout<<"A hat"<<endl;
		std::cout<<this->A_hat->get_output_shape(0) << " " << this->A_hat-> get_output_shape(1)<<endl;
		std::cout<<"D hat"<<endl;
		std::cout<<this->D_hat->get_output_shape(0) << " " << this->D_hat-> get_output_shape(1)<<endl;

		this->output = op::matmul(this->D_hat, op::matmul(this -> A_hat, op::matmul(this -> X, this-> weight)));
		this->output->eval();
		std::cout<<"output"<<endl;
		std::cout<<this->output->get_output_shape(0) << " " << this->output-> get_output_shape(1)<<endl;

	}

    template<typename T>
	unsigned int GCNDecoder<T>::get_num_params() {return this->weight->get_output_size();}

    template class GCNDecoder<int>;
	template class GCNDecoder<float>;
	template class GCNDecoder<double>;

	template<typename T>
	GCNDecoder<T> *gcn_decoder(vector <int> train_pos_u, vector <int> train_pos_v, 
                    vector <int> train_neg_u, vector <int> train_neg_v, op::Operation<T> *X) {
		return new GCNDecoder<T>(train_pos_u, train_pos_v, train_neg_u, train_neg_v, *X);
	}

	template GCNDecoder<int> *gcn_decoder(vector <int> train_pos_u, vector <int> train_pos_v, 
                    vector <int> train_neg_u, vector <int> train_neg_v, op::Operation<int> *X);
	template GCNDecoder<float> *gcn_decoder(vector <int> train_pos_u, vector <int> train_pos_v, 
                    vector <int> train_neg_u, vector <int> train_neg_v, op::Operation<float> *X);
	template GCNDecoder<double> *gcn_decoder(vector <int> train_pos_u, vector <int> train_pos_v, 
                    vector <int> train_neg_u, vector <int> train_neg_v, op::Operation<double> *X);

}
}