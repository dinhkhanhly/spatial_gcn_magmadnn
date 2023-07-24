/**
 * @file gcn_layer.h
 *
 * It functions as a FullyConnected layer, only it
 * does not have a bias.
 */

#pragma once
#include "tensor/tensor.h"
#include "layer/layer.h"
#include "compute/operation.h"
#include "compute/tensor_operations.h"

namespace magmadnn {
namespace layer {

	template<typename T>
	class GCNLayer: public Layer<T> {
	
		public:
			GCNLayer(op::Operation<T> *A_hat, op::Operation<T> *D_hat, op::Operation<T> *X);
			virtual ~GCNLayer();
			std::vector<op::Operation<T> *> get_weights();

			unsigned int get_num_params();


		protected:
			void init();
			op::Operation<T> *A_hat;
			op::Operation<T> *D_hat;
			op::Operation<T> *X;

			Tensor<T> *wTensor;

			op::Operation<T> *weight;

	};


	template<typename T>
	GCNLayer<T> *gcn_layer(op::Operation<T> *A_hat, op::Operation<T> *D_hat, op::Operation<T> *X);

}
}


