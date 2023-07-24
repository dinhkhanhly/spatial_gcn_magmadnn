/**
 * @file layers.h
 * @author Daniel Nichols
 * @version 0.1
 * @date 2019-02-27
 *
 * @copyright Copyright (c) 2019
 */
#pragma once

/* Load in all the layer headers */

#include "activation/activationlayer.h"
#include "batchnorm/batchnormlayer.h"
#include "concat/concatlayer.h"
#include "conv2d/conv2dlayer.h"
#include "conv2dtranspose/conv2dtransposelayer.h"
#include "dropout/dropoutlayer.h"
#include "flatten/flattenlayer.h"
#include "fullyconnected/fullyconnectedlayer.h"
#include "globalavgpooling/globalavgpooling.h"
#include "input/inputlayer.h"
#include "lstm/lstm.h"
#include "output/outputlayer.h"
#include "pooling/poolinglayer.h"
#include "shortcut/shortcutlayer.h"
#include "simplelstm/simplelstm.h"
#include "layer/wrapper.h"
#include "awesome/awesome.h"
#include "gcn/gcn_layer.h"