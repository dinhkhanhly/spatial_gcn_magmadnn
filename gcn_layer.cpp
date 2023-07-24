/**
* @file gcn_layer.cpp 
*/

#include"layer/gcn/gcn_layer.h"
#include "tensor/tensor_io.h"

using namespace std;

namespace magmadnn {
namespace layer {
        
    template<typename T>
	GCNLayer<T>::GCNLayer(op::Operation<T> *A_hat, op::Operation<T> *D_hat, op::Operation<T> *X)
	:Layer<T>::Layer(X->get_output_shape(), X),A_hat(A_hat), D_hat(D_hat), X(X) {		
		init();
	}
    template<typename T>
	GCNLayer<T>::~GCNLayer() {
		delete this->weight;
	}
	
	template<typename T>
	std::vector<op::Operation<T> *> GCNLayer<T>::get_weights() {
		return {this->weight};
	}

    template<typename T>
	void GCNLayer<T>::init() {
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
	unsigned int GCNLayer<T>::get_num_params() {return this->weight->get_output_size();}

    template class GCNLayer<int>;
	template class GCNLayer<float>;
	template class GCNLayer<double>;

	template<typename T>
	GCNLayer<T> *gcn_layer(op::Operation<T> *A_hat, op::Operation<T> *D_hat, op::Operation<T> *X) {
		return new GCNLayer<T>(A_hat, D_hat, X);
	}

	template GCNLayer<int> *gcn_layer(op::Operation<int> *A_hat, op::Operation<int> *D_hat, op::Operation<int> *X);
	template GCNLayer<float> *gcn_layer(op::Operation<float> *A_hat, op::Operation<float> *D_hat, op::Operation<float> *X);
	template GCNLayer<double> *gcn_layer(op::Operation<double> *A_hat, op::Operation<double> *D_hat, 
	op::Operation<double> *X);

}
}