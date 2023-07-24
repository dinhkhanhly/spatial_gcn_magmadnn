/**
 * @file gcn_decoder.h
 *
 * It decodes gcn for link prediction purpose.
 */

#pragma once
#include "tensor/tensor.h"
#include "layer/layer.h"
#include "compute/operation.h"
#include "compute/tensor_operations.h"

namespace magmadnn {
namespace layer {

	template<typename T>
	class GCNDecoder: public Layer<T> {
	
		public:
			GCNDecoder(vector <int> train_pos_u, vector <int> train_pos_v, 
                    vector <int> train_neg_u, vector <int> train_neg_v, op::Operation<T> *X);
			virtual ~GCNDecoder();
			std::vector<op::Operation<T> *> get_weights();

			unsigned int get_num_params();
            // vector <int> train_pos_u(TRAIN_SIZE), train_pos_v(TRAIN_SIZE), train_neg_u(TRAIN_SIZE), train_neg_v(TRAIN_SIZE);



		protected:
			void init();
			vector <int> train_pos_u;
            vector <int> train_pos_v;
            vector <int> train_neg_u;
            vector <int> train_neg_v;
			op::Operation<T> *X;

	};


	template<typename T>
	GCNDecoder<T> *gcn_decoder(vector <int> train_pos_u, vector <int> train_pos_v, 
                    vector <int> train_neg_u, vector <int> train_neg_v, op::Operation<T> *X);

}
}


