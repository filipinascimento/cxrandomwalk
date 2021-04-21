
#define PY_SSIZE_T_CLEAN
#include "PyCXVersion.h"
#include <CVDistribution.h>
#include <CVNetwork.h>
#include <CVSet.h>
#include <Python.h>
#include <pthread.h>

#include "structmember.h"

// #define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cxrandomwalk_ARRAY_API
#include <numpy/arrayobject.h>

#if CV_USE_OPENMP
#include <omp.h>
#endif //_OPENMP

static PyArrayObject *
pyvector(PyObject *objin)
{
	return (PyArrayObject *)PyArray_ContiguousFromObject(objin, NPY_FLOAT, 1, 1);
}

static PyArrayObject *
convertToUIntegerArray(PyObject *object, int minDepth, int maxDepth)
{
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(
		object, PyArray_DescrFromType(NPY_UINT64), minDepth, maxDepth, flags, NULL);
}
static PyArrayObject *
convertToIntegerArray(PyObject *object, int minDepth, int maxDepth)
{
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(
		object, PyArray_DescrFromType(NPY_INT64), minDepth, maxDepth, flags, NULL);
}
static PyArrayObject *
convertToDoubleArray(PyObject *object, int minDepth, int maxDepth)
{
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(object,
							 PyArray_DescrFromType(NPY_FLOAT64),
							 minDepth,
							 maxDepth,
							 flags,
							 NULL);
}
static PyArrayObject *
convertToFloatArray(PyObject *object, int minDepth, int maxDepth)
{
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(object,
							 PyArray_DescrFromType(NPY_FLOAT32),
							 minDepth,
							 maxDepth,
							 flags,
							 NULL);
}

/* ==== Create 1D Carray from PyArray ======================
																																Assumes PyArray
	 is contiguous in memory.             */
static void *
pyvector_to_Carrayptrs(PyArrayObject *arrayin)
{
	int i, n;

	n = arrayin->dimensions[0];
	return PyArray_DATA(arrayin); /* pointer to arrayin data as double */
}

/* ==== Check that PyArrayObject is a double (Float) type and a vector
				 ============== return 1 if an error and raise exception */
static int
not_floatvector(PyArrayObject *vec)
{
	if (vec->descr->type_num != NPY_FLOAT) {
		PyErr_SetString(PyExc_ValueError,
						"In not_floatvector: array must be of "
						"type Float and 1 dimensional (n).");
		return 1;
	}
	return 0;
}

/* ==== Check that PyArrayObject is a double (Float) type and a vector
				 ============== return 1 if an error and raise exception */
// FIXME: make it work for 32bits
static int
not_intvector(PyArrayObject *vec)
{
	if (vec->descr->type_num != NPY_UINT64) {
		PyErr_SetString(
			PyExc_ValueError,
			"In not_intvector: array must be of type Long and 1 dimensional (n).");
		return 1;
	}
	return 0;
}

typedef struct _PyAgent {
	PyObject_HEAD CVNetwork *network;
	CVBool verbose;
} PyAgent;

int PyAgent_traverse(PyAgent *self, visitproc visit, void *arg)
{
	// Py_VISIT(self->...);
	return 0;
}

int PyAgent_clear(PyAgent *self)
{
	// Py_CLEAR(self->...);
	return 0;
}

void PyAgent_dealloc(PyAgent *self)
{
	// PyObject_GC_UnTrack(self);
	// PyAgent_clear(self);
	if (self->network) {
		CVNetworkDestroy(self->network);
	}
	Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *
PyAgent_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyAgent *self;
	self = (PyAgent *)type->tp_alloc(type, 0);
	self->network = NULL;
	return (PyObject *)self;
}

int PyAgent_init(PyAgent *self, PyObject *args, PyObject *kwds){
	CVRandomSeedDev();
	static char *kwlist[] = {
		"vertexCount",
		"edges",
		"directed",
		"weights",
		NULL
	};
	PyObject *edgesObject = NULL;
	PyObject *weightsObject = NULL;
	PyArrayObject *edgesArray = NULL;
	PyArrayObject *weightsArray = NULL;
	Py_ssize_t vertexCount = 0;
	int isDirected = 0;

	if (!PyArg_ParseTupleAndKeywords(args,
									 kwds,
									 "nO|pO",
									 kwlist,
									 &vertexCount,
									 &edgesObject,
									 &isDirected,
									 &weightsObject)) {
		return -1;
	}

	if (vertexCount <= 0) {
		PyErr_SetString(PyExc_TypeError,
						"The number of ndoes (vertexCount) must be a positive integer.");
		return -1;
	}

	if (!(edgesArray = convertToIntegerArray(edgesObject, 1, 2))) {
		// PyErr_SetString(PyExc_TypeError,"Error creating arrays.");
		return -1;
	}

	CVSize edgeCount = (CVSize)PyArray_SIZE(edgesArray) / 2;
	npy_int64 *edges = PyArray_DATA(edgesArray);

	if (weightsObject &&
		!(weightsArray = convertToDoubleArray(weightsObject, 1, 1))) {
		// PyErr_SetString(PyExc_TypeError,"The weights attribute must be a float32
		// numpy array.");
		Py_XDECREF(edgesArray);
		return -1;
	}

	CVSize weightsCount = 0;
	double *weights = NULL;

	if (weightsArray) {
		weightsCount = (CVSize)PyArray_SIZE(weightsArray);
		weights = PyArray_DATA(weightsArray);
	}

	if (weights && weightsCount != edgeCount) {
		PyErr_SetString(
			PyExc_TypeError,
			"Weights should have the same dimension as the number of edges.");
		Py_XDECREF(edgesArray);
		Py_XDECREF(weightsArray);
		return -1;
	}

	self->network = CVNewNetwork(
		vertexCount, weights ? CVTrue : CVFalse, isDirected ? CVTrue : CVFalse);
	for (CVIndex i = 0; i < edgeCount; i++) {
		CVIndex fromIndex = (CVIndex)edges[2 * i];
		CVIndex toIndex = (CVIndex)edges[2 * i + 1];
		CVDouble weight = 1.0;
		if (fromIndex >= vertexCount || toIndex >= vertexCount) {
			PyErr_SetString(
				PyExc_TypeError,
				"Edge indices should not be higher than the number of vertices.");
			Py_XDECREF(edgesArray);
			Py_XDECREF(weightsArray);
			return -1;
		}
		if (weights) {
			weight = weights[i];
		}
		CVNetworkAddNewEdge(self->network, fromIndex, toIndex, weight);
	}

	Py_XDECREF(edgesArray);
	Py_XDECREF(weightsArray);
	return 0;
}

PyMemberDef PyAgent_members[] = {
	// {"attractiveConstant", T_FLOAT, offsetof(PyAgent, attractiveConstant),
	// 0,"Attractive constant"},
	// {"repulsiveConstant", T_FLOAT, offsetof(PyAgent, repulsiveConstant),
	// 0,"Repulsive constant"},
	// {"viscosityConstant", T_FLOAT, offsetof(PyAgent, viscosityConstant),
	// 0,"Viscosity constant"},
	{NULL} /* Sentinel */
};

// PyObject * PyAgent_getEdges(PyAgent *self, void *closure){
// 	// Py_INCREF(self->edgesArray);
// 	return (PyObject*)self->edgesArray;
// }

static PyGetSetDef PyAgent_getsetters[] = {
	// {"edges", (getter) PyAgent_getEdges,  NULL,"Edges array", NULL},
	{NULL} /* Sentinel */
};

PyObject * PyAgent_generateWalks(PyAgent *self, PyObject *args, PyObject *kwds){
	static char *kwlist[] = {
		"p",
		"q",
		"windowSize",
		"walksPerNode",
		"verbose",
		"filename",
		"labels",
		"callback",
		"updateInterval",
		NULL
	};

	float p = 1.0;
	float q = 1.0;
	Py_ssize_t windowSize = 80;
	Py_ssize_t walksPerNode = 80;
	int verbose = 0;
	char* outputPath = NULL;
	PyObject* labels = NULL;
	PyObject* callback = NULL ;
	Py_ssize_t updateInterval = 1000;

	if (!PyArg_ParseTupleAndKeywords(args,
									 kwds,
									 "|ffnnpsOOn",
									 kwlist,
									 &p,
									 &q,
									 &windowSize,
									 &walksPerNode,
									 &verbose,
									 &outputPath,
									 &labels,
									 &callback,
									 &updateInterval)) {
		return NULL;
	}
	if (callback != NULL ) {
		if (!PyCallable_Check(callback )) {
			PyErr_SetString(PyExc_ValueError, "Invalid callback.") ;
		return NULL ;
		}
	}



	FILE *outputFile = NULL;
	if(outputPath){
		outputFile = fopen(outputPath, "w");
		if (!outputFile) {
			PyErr_Format(PyExc_FileNotFoundError,"Cannot save to file \"%s\". \n", outputPath);
			return NULL;
		}
	}




	CVNetwork *network = self->network;

	CVSize verticesCount = network->verticesCount;
	CVSize sentencesCount = network->verticesCount * walksPerNode;
	CVIndex *sentences = calloc(sentencesCount * windowSize,
								sizeof(CVIndex)); // all indices are shifted by 1


	CVString* labelsData = NULL;
	CVSize labelsDataSize = 0;
	if(labels){
		Py_ssize_t numLines = PyList_Size(labels);
		if (numLines >= verticesCount){
			labelsData = calloc(numLines,sizeof(CVString));
			for (CVIndex index=0; index<numLines; index++){
				labelsDataSize = numLines;
				PyObject* strObj = PyList_GetItem(labels, index);
				if (PyUnicode_Check(strObj)) {
					PyObject * temp_bytes = PyUnicode_AsEncodedString(strObj, "UTF-8", "strict"); // Owned reference
					if (temp_bytes != NULL) {
							labelsData[index] = CVNewStringFromString(PyBytes_AS_STRING(temp_bytes)); // Borrowed pointer
							// printf("\n%"CVIndexScan": %s\n",index,labelsData[index]);
							Py_DECREF(temp_bytes);
					}
				}
			}
		}
	}



	unsigned int *seeds = calloc(sentencesCount, sizeof(unsigned int));

	unsigned int initialSeed = (unsigned int)time(NULL);
	for (CVIndex sentenceIndex = 0; sentenceIndex < sentencesCount;
		 sentenceIndex++) {
		seeds[sentenceIndex] = rand_r(&initialSeed) ^ (unsigned int)sentenceIndex;
	}

	CVInteger *currentProgress = calloc(1, sizeof(CVInteger));
	CVInteger *shallStop = calloc(1, sizeof(CVInteger));
	*shallStop = 0;

	CVParallelForStart(distributionsLoop, sentenceIndex, sentencesCount){

		if(!*shallStop){
			if (CVAtomicIncrementInteger(currentProgress) % updateInterval == 0) {
				CVParallelLoopCriticalRegionStart(distributionsLoop){
					if(verbose){
						printf("Walks: %" CVIndexScan "/%" CVIndexScan
								" (%.2f%%)                                                      "
								"           \r",
								(CVIndex)(*currentProgress),
								sentencesCount,
								(*currentProgress) / (float)(sentencesCount - 1) * 100.0);
						fflush(stdout);
					}

					if (PyErr_CheckSignals() != 0){
						*shallStop = 1;
						printf("Stopping Walks                                \n");
						fflush(stdout);
					}else if(callback){
							PyObject* pArgs = Py_BuildValue( "nn",(Py_ssize_t)(*currentProgress),(Py_ssize_t)sentencesCount) ;
							PyObject* pKywdArgs = NULL ;
							PyObject_Call( callback, pArgs, NULL ) ;
							Py_DECREF( pArgs);
					}

				}
				CVParallelLoopCriticalRegionEnd(distributionsLoop);
			}
		}
		if(!*shallStop){
			CVIndex currentNode = sentenceIndex % verticesCount;
			CVIndex previousNode = currentNode;
			CVUIntegerSet *previousNeighborsSet = CVNewUIntegerSet();
			unsigned int *seedRef = seeds + sentenceIndex;
			sentences[sentenceIndex * windowSize + 0] =
				currentNode + 1; // Always shifted by 1;
			if (q == 1.0 && p == 1.0) {
				for (CVIndex walkStep = 1; walkStep < windowSize; walkStep++) { //
					CVIndex *neighbors = network->vertexEdgesLists[currentNode];
					CVIndex neighborCount = network->vertexNumOfEdges[currentNode];
					CVIndex *neighEdges = network->vertexEdgesIndices[currentNode];
					if (neighborCount > 0) {
						CVFloat *probabilities = calloc(neighborCount, sizeof(CVFloat));
						for (CVIndex neighIndex = 0; neighIndex < neighborCount;
							neighIndex++) {
							CVIndex edgeIndex = neighEdges[neighIndex];
							CVFloat weight = 1.0;

							if (network->edgeWeighted) {
								weight = network->edgesWeights[edgeIndex];
							}
							probabilities[neighIndex] = weight;
						}

						CVDouble choice = ((double)rand_r(seedRef) / RAND_MAX);
						CVDistribution *distribution =
							CVCreateDistribution(probabilities, NULL, neighborCount);

						previousNode = currentNode;
						currentNode =
							neighbors[CVDistributionIndexForChoice(distribution, choice)];
						sentences[sentenceIndex * windowSize + walkStep] =
							currentNode + 1; // Always shifted by 1;
						CVDestroyDistribution(distribution);
						free(probabilities);
					}
					else {
						break;
					}
				}
			}
			else {
				for (CVIndex walkStep = 1; walkStep < windowSize; walkStep++) { //
					CVIndex *neighbors = network->vertexEdgesLists[currentNode];
					CVIndex neighborCount = network->vertexNumOfEdges[currentNode];
					CVIndex *neighEdges = network->vertexEdgesIndices[currentNode];
					if (neighborCount > 0) {
						CVFloat *probabilities = calloc(neighborCount, sizeof(CVFloat));
						for (CVIndex neighIndex = 0; neighIndex < neighborCount;
							neighIndex++) {
							CVIndex edgeIndex = neighEdges[neighIndex];
							CVIndex candidateIndex = neighbors[neighIndex];
							CVFloat weight = 1.0;

							if (network->edgeWeighted) {
								weight = network->edgesWeights[edgeIndex];
							}

							if (neighbors[neighIndex] == previousNode) {
								probabilities[neighIndex] = weight * 1 / p;
							}
							else if (CVUIntegerSetHas(previousNeighborsSet, candidateIndex)) {
								probabilities[neighIndex] = weight;
							}
							else {
								probabilities[neighIndex] = weight * 1 / q;
							}
						}

						CVDouble choice = ((double)rand_r(seedRef) / RAND_MAX);
						CVDistribution *distribution =
							CVCreateDistribution(probabilities, NULL, neighborCount);

						previousNode = currentNode;
						currentNode =
							neighbors[CVDistributionIndexForChoice(distribution, choice)];
						sentences[sentenceIndex * windowSize + walkStep] =
							currentNode + 1; // Always shifted by 1;
						CVDestroyDistribution(distribution);
						free(probabilities);

						CVUIntegerSetClear(previousNeighborsSet);
						for (CVIndex neighIndex = 0; neighIndex < neighborCount;
							neighIndex++) {
							CVUIntegerSetAdd(previousNeighborsSet, neighbors[neighIndex]);
						}
					}
					else {
						break;
					}
				}
			}
			CVUIntegerSetDestroy(previousNeighborsSet);
		}
	}
	CVParallelForEnd(distributionsLoop);
	

	free(currentProgress);
	
	if(*shallStop){
		printf("Stopped                                \n");
		// free(sentences);
		// free(shallStop);
		// PyErr_Format(PyExc_FileNotFoundError,"Error happened.");

		return NULL;
	}

	free(shallStop);
	
	if(verbose){
		printf("DONE                                \n");
	}

	PyListObject* sentencesList = NULL;
	if(!outputFile){
		sentencesList = PyList_New(sentencesCount);
	}
	for (CVIndex sentenceIndex = 0; sentenceIndex < sentencesCount; sentenceIndex++) {
		PyListObject* walkList = NULL;
		if(!outputFile){
			walkList = PyList_New(0);
			PyList_SET_ITEM(sentencesList,sentenceIndex,walkList);
		}
		for (CVIndex walkStep = 0; walkStep < windowSize; walkStep++) {
			CVIndex nodeIndexWithOffset = sentences[sentenceIndex * windowSize + walkStep];
			if (nodeIndexWithOffset > 0) {
				if(outputFile){
					if(labelsData){
						fprintf(outputFile,"%s ",labelsData[(nodeIndexWithOffset-1)]);
						// printf("\n%d: %s\n",(int)(nodeIndexWithOffset-1),labelsData[nodeIndexWithOffset-1]);
					}else{
						fprintf(outputFile,"%"CVUIntegerScan" ",(nodeIndexWithOffset-1));
					}
				}else{
					if(labelsData){
						PyObject* value = Py_BuildValue("s", labelsData[(nodeIndexWithOffset-1)]);
						// printf("\n%d: %s\n",(int)(nodeIndexWithOffset-1),labelsData[nodeIndexWithOffset-1]);
						PyList_Append(walkList, value);
						Py_DECREF(value);
					}else{
						PyObject* value = PyLong_FromUnsignedLong((unsigned long)(nodeIndexWithOffset-1));
						PyList_Append(walkList, value);
						Py_DECREF(value);
					}
				}
				// printf("%" CVUIntegerScan " ", (nodeIndexWithOffset - 1));
			}
			else {
				break;
			}
		}
		if(outputFile){
			fprintf(outputFile,"\n");
		}
	}
	
	free(sentences);
	if(labelsData){
		for (CVIndex index=0; index<labelsDataSize; index++){
			CVStringDestroy(labelsData[index]);
		}
		free(labelsData);
	}

	if(outputFile){
		Py_RETURN_NONE;
	}else{
		return (PyObject*)sentencesList;
	}
}

// PyObject* PyAgent_isRunning(PyAgent *self, PyObject *Py_UNUSED(ignored)){
// 	return Py_BuildValue("O", self->_isRunning ? Py_True: Py_False);
// }

// PyObject* PyAgent_startLayout(PyAgent *self, PyObject *Py_UNUSED(ignored)){
// 	if(!self->_isRunning){
// 		self->_isRunning = CVTrue;
// 		pthread_create(&(self->_thread), NULL, (void*)_threadIterate,
// self);
// 	}
// 	Py_RETURN_NONE;
// }

// PyObject* PyAgent_stopLayout(PyAgent *self, PyObject *Py_UNUSED(ignored)){

// 	if(self->_isRunning){
// 		self->_shallStop=CVTrue;
// 		pthread_join(self->_thread, NULL);
// 		self->_isRunning=CVFalse;
// 		self->_shallStop=CVFalse;
// 	}
// 	Py_RETURN_NONE;
// }

static PyMethodDef PyAgent_methods[] = {
	// {"start", (PyCFunction) PyAgent_startLayout, METH_NOARGS,
	//  "Starts the layout algorithm"
	// },
	// {"stop", (PyCFunction) PyAgent_stopLayout, METH_NOARGS,
	//  "Stops the layout algorithm"
	// },
	// {"running", (PyCFunction) PyAgent_isRunning, METH_NOARGS,
	//  "Returns True if it is running"
	// },
	{"generateWalks",
	 (PyCFunction)PyAgent_generateWalks,
	 METH_VARARGS | METH_KEYWORDS,
	 "Create a sequence of walks."},
	{NULL} /* Sentinel */
};

static PyTypeObject PyAgentType = {
	PyVarObject_HEAD_INIT(NULL, 0).tp_name = "cxrandomwalk.Agent",
	.tp_doc = "PyAgent objects",
	.tp_basicsize = sizeof(PyAgent),
	.tp_itemsize = 0,
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, // | Py_TPFLAGS_HAVE_GC,
	.tp_new = PyAgent_new,
	.tp_init = (initproc)PyAgent_init,
	.tp_dealloc = (destructor)PyAgent_dealloc,
	.tp_traverse = NULL, //(traverseproc) PyAgent_traverse,
	.tp_clear = NULL,	 //(inquiry) PyAgent_clear,
	.tp_members = PyAgent_members,
	.tp_methods = PyAgent_methods,
	.tp_getset = PyAgent_getsetters,
};

char cxrandomwalkmod_docs[] = "This is CXRandomWalk module.";

static PyModuleDef cxrandomwalk_mod = {PyModuleDef_HEAD_INIT,
										 .m_name = "cxrandomwalk",
										 .m_doc = cxrandomwalkmod_docs,
										 .m_size = -1,
										 .m_methods = NULL,
										 .m_slots = NULL,
										 .m_traverse = NULL,
										 .m_clear = NULL,
										 .m_free = NULL};

PyMODINIT_FUNC
PyInit_cxrandomwalk(void)
{
	import_array();

	PyObject *m;
	if (PyType_Ready(&PyAgentType) < 0) {
		return NULL;
	}
	m = PyModule_Create(&cxrandomwalk_mod);
	if (m == NULL) {
		return NULL;
	}
	Py_INCREF(&PyAgentType);
	if (PyModule_AddObject(m, "Agent", (PyObject *)&PyAgentType) < 0) {
		Py_DECREF(&PyAgentType);
		Py_DECREF(m);
		return NULL;
	}

	if (PyModule_AddStringConstant(m,"__version__",CVTOKENTOSTRING(k_PYCXVersion))<0) {
			Py_DECREF(m);
			return NULL;
	}

	return m;
}
