/**
 * @file node_pair_multiply.cpp
 */
#if defined(MAGMADNN_CMAKE_BUILD)
#include "magmadnn/config.h"
#endif
#include "compute/node_pair_multiply/node_pair_multiply.h"

namespace magmadnn {
namespace op {

template <typename T>
NodePairMultiply<T>::NodePairMultiply(vector <int> train_pos_u, vector <int> train_pos_v, 
                    vector <int> train_neg_u, vector <int> train_neg_v, Operation<T> *X, bool copy, bool needs_grad)
    : Operation<T>::Operation({train_pos_u, train_pos_v, train_neg_u, train_neg_v, X}, needs_grad), 
                    train_pos_u(train_pos_u), train_pos_v(train_pos_v), train_neg_u(train_neg_u), train_neg_v(train_neg_v),
                    copy(copy) {
    // assert(a->get_memory_type() == b->get_memory_type());
    // assert(a->get_output_size() == b->get_output_size() || a->get_output_size() == 1 || b->get_output_size() == 1);

    
    int num_pos_edges = train_pos_u.size();
    int num_neg_edges = train_neg_u.size();
    int number_of_edges = num_pos_edges + num_neg_edges;
    
    this -> output_shape = {1, number_of_edges};
    this->mem_type = X->get_memory_type();
    this->name = "NodePairMultiply";

    /* Go ahead and create copy tensor if we can */
    this->output_tensor = new Tensor<T>(this->output_shape, {NONE, {}}, this->mem_type);
}

template <typename T>
Tensor<T> *NodePairMultiply<T>::_eval(bool recompute) {
    // std::cout << "[NodePairMultiply<T>::_eval]" << std::endl;

    X_tensor = X->eval(recompute);
    // return this->output_tensor;

    // std::cout << "[AddOp<T>::_eval] a size = " << a_tensor->get_size() << std::endl;
    // std::cout << "[AddOp<T>::_eval] b size = " << b_tensor->get_size() << std::endl;
    // std::cout << "[AddOp<T>::_eval] output size = " << this->output_tensor->get_size() << std::endl;

//     if (a_tensor->get_size() == 1) {
//         a_tensor->get_memory_manager()->sync(true);
//         if (this->output_tensor->get_memory_type() == HOST) {
//             internal::tensor_scalar_add_full_cpu(a_tensor->get(0), b_tensor, this->output_tensor);
//         }
// #if defined(MAGMADNN_HAVE_CUDA)
//         else {
//             internal::tensor_scalar_add_full_device(this->get_custream(), a_tensor->get(0), b_tensor,
//                                                     this->output_tensor);
//             if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
//         }
// #endif
//     } else if (b_tensor->get_size() == 1) {
//         b_tensor->get_memory_manager()->sync(true);
//         if (this->output_tensor->get_memory_type() == HOST) {
//             internal::tensor_scalar_add_full_cpu(b_tensor->get(0), a_tensor, this->output_tensor);
//         }
// #if defined(MAGMADNN_HAVE_CUDA)
//         else {
//             internal::tensor_scalar_add_full_device(this->get_custream(), b_tensor->get(0), a_tensor,
//                                                     this->output_tensor);
//             if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
//         }
// #endif
//     } else {
//         if (this->output_tensor->get_memory_type() == HOST) {
//             internal::geadd_full_cpu((T) 1, a_tensor, (T) 1, b_tensor, this->output_tensor);
//         }
// #if defined(MAGMADNN_HAVE_CUDA)
//         else {
//             // internal::tensor_scalar_add_full_device(
//             //       this->get_custream(),
//             //       T(1.0), b_tensor, this->output_tensor);

//             internal::geadd_full_device(this->get_custream(), (T) 1, a_tensor, (T) 1, b_tensor, this->output_tensor);
//             if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
//         }
// #endif

    int num_pos_edges = train_pos_u.size();
    int num_neg_edges = train_neg_u.size();
    int number_of_edges = num_pos_edges + num_neg_edges;
    int u, v;
    int number_of_features = X->get_output_shape(1);
    float sum, feat1, feat2;
    if (this->output_tensor->get_memory_type() == HOST) {
        for (int i=0;i<num_pos_edges;i++) {
            sum = 0;
            u = train_pos_u[i];
            v = train_pos_v[i];
            for (int j=0;j<number_of_features;j++) {
                feat1 = X -> get_output_tensor() -> get({u, j});
                feat2 = X -> get_output_tensor() -> get({v, j});
                sum += feat1 * feat2;
            }
            this-> output_tensor -> set({0, i}, sum);
        }
    }
// #if defined(MAGMADNN_HAVE_CUDA)
//     else {
//         // internal::tensor_scalar_add_full_device(this->get_custream(), b_tensor->get(0), a_tensor,
//         //                                         this->output_tensor);
//         // if (!this->get_async()) cudaStreamSynchronize(this->get_custream());
//         for (int i=0;i<num_pos_edges;i++) {
//             sum = 0;
//             u = train_pos_u[i];
//             v = train_pos_v[i];
//             for (int j=0;j<number_of_features;j++) {
//                 feat1 = X -> get_output_tensor() -> get({u, j});
//                 feat2 = X -> get_output_tensor() -> get({v, j});
//                 sum += feat1 * feat2;
//             }
//             this-> output_tensor -> set({0, i}, sum);
//         }
//     }
// #endif

    return this->output_tensor;
}

template <typename T>
Tensor<T> *AddOp<T>::_grad(Operation<T> *consumer, Operation<T> *var, Tensor<T> *grad) {
    // this->_grad_cache[(uintptr_t) var] = grad;
    // std::cout << "[AddOp<T>::_grad]" << std::endl;
    // std::cout << "[AddOp<T>::_grad] grad = " << grad << std::endl;
    return grad;
}
template class AddOp<int>;
template class AddOp<float>;
template class AddOp<double>;

template <typename T>
AddOp<T> *add(Operation<T> *a, Operation<T> *b, bool copy, bool needs_grad) {
    return new AddOp<T>(a, b, copy, needs_grad);
}
template AddOp<int> *add(Operation<int> *a, Operation<int> *b, bool copy, bool needs_grad);
template AddOp<float> *add(Operation<float> *a, Operation<float> *b, bool copy, bool needs_grad);
template AddOp<double> *add(Operation<double> *a, Operation<double> *b, bool copy, bool needs_grad);

}  // namespace op
}  // namespace magmadnn
}