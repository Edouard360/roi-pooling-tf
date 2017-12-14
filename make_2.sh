TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

CXXFLAGS=''

if [[ "$OSTYPE" =~ ^darwin ]]; then
	CXXFLAGS+='-undefined dynamic_lookup'
fi

cd module_2

g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
	-I $TF_INC -fPIC $CXXFLAGS -D_GLIBCXX_USE_CXX11_ABI=0 #-L$TF_LIB -ltensorflow_framework

g++ -std=c++11 -shared -o roi_pooling_op_grad.so roi_pooling_op_grad.cc \
	-I $TF_INC -fPIC $CXXFLAGS -D_GLIBCXX_USE_CXX11_ABI=0 #-L$TF_LIB -ltensorflow_framework

cd ..