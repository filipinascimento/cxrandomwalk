
#define PY_SSIZE_T_CLEAN
#include <CVDistribution.h>
#include <CVNetwork.h>
#include <CVNetworkCentrality.h>
#include <CVSet.h>
#include <Python.h>

#include "PyCXVersion.h"
// #include <pthread.h>

#include "structmember.h"

// #define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cxrandomwalk_core_ARRAY_API
#include <numpy/arrayobject.h>

#if CV_USE_OPENMP
#	include <omp.h>
#endif //_OPENMP

CV_INLINE CVDouble getRandomChoice(unsigned int *seedRef) {
#ifdef __WIN32__
	unsigned int randomNumber;
	rand_s(&randomNumber);
	return ((double)randomNumber / UINT_MAX);
#else
	return ((double)rand_r(seedRef) / RAND_MAX);
#endif
}

static PyArrayObject *pyvector(PyObject *objin) {
	return (PyArrayObject *)PyArray_ContiguousFromObject(objin, NPY_FLOAT, 1, 1);
}

static PyArrayObject *convertToUIntegerArray(PyObject *object, int minDepth, int maxDepth) {
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(
		object, PyArray_DescrFromType(NPY_UINT64), minDepth, maxDepth, flags, NULL);
}

static PyArrayObject *convertToIntegerArray(PyObject *object, int minDepth, int maxDepth) {
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(
		object, PyArray_DescrFromType(NPY_INT64), minDepth, maxDepth, flags, NULL);
}

static PyArrayObject *convertToDoubleArray(PyObject *object, int minDepth, int maxDepth) {
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(
		object, PyArray_DescrFromType(NPY_FLOAT64), minDepth, maxDepth, flags, NULL);
}

static PyArrayObject *convertToFloatArray(PyObject *object, int minDepth, int maxDepth) {
	int flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
	return PyArray_FromAny(
		object, PyArray_DescrFromType(NPY_FLOAT32), minDepth, maxDepth, flags, NULL);
}

/* ==== Create 1D Carray from PyArray ======================
Assumes PyArray
is contiguous in memory.             */
static void *pyvector_to_Carrayptrs(PyArrayObject *arrayin) {
	int i, n;

	n = arrayin->dimensions[0];
	return PyArray_DATA(arrayin); /* pointer to arrayin data as double */
}

/* ==== Check that PyArrayObject is a double (Float) type and a vector
============== return 1 if an error and raise exception */
static int not_floatvector(PyArrayObject *vec) {
	if (vec->descr->type_num != NPY_FLOAT) {
		PyErr_SetString(PyExc_ValueError, "In not_floatvector: array must be of "
										  "type Float and 1 dimensional (n).");
		return 1;
	}
	return 0;
}

/* ==== Check that PyArrayObject is a double (Float) type and a vector
============== return 1 if an error and raise exception */
// FIXME: make it work for 32bits
static int not_intvector(PyArrayObject *vec) {
	if (vec->descr->type_num != NPY_UINT64) {
		PyErr_SetString(PyExc_ValueError, "In not_intvector: array must be of "
										  "type Long and 1 dimensional (n).");
		return 1;
	}
	return 0;
}

typedef struct _PyAgent {
	PyObject_HEAD CVNetwork *network;
	CVBool verbose;
} PyAgent;

int PyAgent_traverse(PyAgent *self, visitproc visit, void *arg) {
	// Py_VISIT(self->...);
	return 0;
}

int PyAgent_clear(PyAgent *self) {
	// Py_CLEAR(self->...);
	return 0;
}

void PyAgent_dealloc(PyAgent *self) {
	// PyObject_GC_UnTrack(self);
	// PyAgent_clear(self);
	if (self->network) {
		CVNetworkDestroy(self->network);
	}
	Py_TYPE(self)->tp_free((PyObject *)self);
}

PyObject *PyAgent_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
	PyAgent *self;
	self = (PyAgent *)type->tp_alloc(type, 0);
	self->network = NULL;
	return (PyObject *)self;
}

int PyAgent_init(PyAgent *self, PyObject *args, PyObject *kwds) {
	CVRandomSeedDev();
	static char *kwlist[] = {"vertexCount", "edges", "directed", "weights", NULL};
	PyObject *edgesObject = NULL;
	PyObject *weightsObject = NULL;
	PyArrayObject *edgesArray = NULL;
	PyArrayObject *weightsArray = NULL;
	Py_ssize_t vertexCount = 0;
	int isDirected = 0;

	if (!PyArg_ParseTupleAndKeywords(
			args, kwds, "nO|pO", kwlist, &vertexCount, &edgesObject, &isDirected, &weightsObject)) {
		return -1;
	}

	if (vertexCount <= 0) {
		PyErr_SetString(
			PyExc_TypeError, "The number of ndoes (vertexCount) must be a positive integer.");
		return -1;
	}

	if (!(edgesArray = convertToIntegerArray(edgesObject, 1, 2))) {
		// PyErr_SetString(PyExc_TypeError,"Error creating arrays.");
		return -1;
	}

	CVSize edgeCount = (CVSize)PyArray_SIZE(edgesArray) / 2;
	npy_int64 *edges = PyArray_DATA(edgesArray);

	if (weightsObject && !(weightsArray = convertToDoubleArray(weightsObject, 1, 1))) {
		// PyErr_SetString(PyExc_TypeError,"The weights attribute must be a
		// float32 numpy array.");
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
			PyExc_TypeError, "Weights should have the same dimension as the number of edges.");
		Py_XDECREF(edgesArray);
		Py_XDECREF(weightsArray);
		return -1;
	}

	self->network =
		CVNewNetwork(vertexCount, weights ? CVTrue : CVFalse, isDirected ? CVTrue : CVFalse);
	for (CVIndex i = 0; i < edgeCount; i++) {
		CVIndex fromIndex = (CVIndex)edges[2 * i];
		CVIndex toIndex = (CVIndex)edges[2 * i + 1];
		CVDouble weight = 1.0;
		if (fromIndex >= vertexCount || toIndex >= vertexCount) {
			PyErr_SetString(PyExc_TypeError, "Edge indices should not be higher than the "
											 "number of vertices.");
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

PyObject *PyAgent_generateWalks(PyAgent *self, PyObject *args, PyObject *kwds) {
	static char *kwlist[] = {"nodes", "p", "q", "windowSize", "walksPerNode", "verbose", "filename",
		"labels", "callback", "updateInterval", NULL};

	float p = 1.0;
	float q = 1.0;
	Py_ssize_t windowSize = 80;
	Py_ssize_t walksPerNode = 80;
	int verbose = 0;
	char *outputPath = NULL;
	PyObject *labels = NULL;
	PyObject *nodes = NULL;
	PyObject *callback = NULL;
	Py_ssize_t updateInterval = 1000;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OffnnpsOOn", kwlist, &nodes, &p, &q, &windowSize,
			&walksPerNode, &verbose, &outputPath, &labels, &callback, &updateInterval)) {
		return NULL;
	}
	if (callback != NULL) {
		if (!PyCallable_Check(callback)) {
			PyErr_SetString(PyExc_ValueError, "Invalid callback.");
			return NULL;
		}
	}

	FILE *outputFile = NULL;
	if (outputPath) {
		outputFile = fopen(outputPath, "w");
		if (!outputFile) {
			PyErr_Format(PyExc_FileNotFoundError, "Cannot save to file \"%s\". \n", outputPath);
			return NULL;
		}
	}

	CVNetwork *network = self->network;
	windowSize += 1; // Our window size is in number of nodes, but users provide
					 // number of hops.

	CVSize verticesCount = network->verticesCount;
	CVIndex *nodesArray = NULL;
	CVSize nodesArraySize = 0;
	// if nodes is not specified, put all nodes in the array
	if (!nodes) {
		nodesArray = calloc(verticesCount, sizeof(CVIndex));
		for (CVIndex index = 0; index < verticesCount; index++) {
			nodesArray[index] = index;
		}
		nodesArraySize = verticesCount;
	} else {
		PyArrayObject *nodesArrayObject = NULL;
		if (!(nodesArrayObject = convertToUIntegerArray(nodes, 1, 1))) {
			PyErr_SetString(PyExc_TypeError, "Error creating arrays.");
			return NULL;
		}
		nodesArraySize = (CVSize)PyArray_SIZE(nodesArrayObject);
		nodesArray = PyArray_DATA(nodesArrayObject);
	}
	// check if indices are valid
	for (CVIndex index = 0; index < nodesArraySize; index++) {
		if (nodesArray[index] >= verticesCount) {
			free(nodesArray);
			PyErr_SetString(PyExc_TypeError, "Node indices should not be higher than the "
											 "number of vertices.");
			return NULL;
		}
	}

	CVSize sentencesCount = nodesArraySize * walksPerNode;
	CVIndex *sentences = calloc(sentencesCount * windowSize,
		sizeof(CVIndex)); // all indices are shifted by 1

	CVString *labelsData = NULL;
	CVSize labelsDataSize = 0;
	if (labels) {
		Py_ssize_t numLines = PyList_Size(labels);
		if (numLines >= verticesCount) {
			labelsData = calloc(numLines, sizeof(CVString));
			for (CVIndex index = 0; index < numLines; index++) {
				labelsDataSize = numLines;
				PyObject *strObj = PyList_GetItem(labels, index);
				if (PyUnicode_Check(strObj)) {
					PyObject *temp_bytes =
						PyUnicode_AsEncodedString(strObj, "UTF-8", "strict"); // Owned reference
					if (temp_bytes != NULL) {
						labelsData[index] = CVNewStringFromString(
							PyBytes_AS_STRING(temp_bytes)); // Borrowed pointer
						// printf("\n%"CVIndexScan":
						// %s\n",index,labelsData[index]);
						Py_DECREF(temp_bytes);
					}
				}
			}
		}
	}

	unsigned int *seeds = calloc(sentencesCount, sizeof(unsigned int));

	unsigned int initialSeed = (unsigned int)time(NULL);
	for (CVIndex sentenceIndex = 0; sentenceIndex < sentencesCount; sentenceIndex++) {
#ifdef __WIN32__
		unsigned int randomNumber;
		rand_s(&randomNumber);
		randomNumber ^= (unsigned int)sentenceIndex;
		seeds[sentenceIndex] = randomNumber;
#else
		seeds[sentenceIndex] = rand_r(&initialSeed) ^ (unsigned int)sentenceIndex;
#endif
	}

	CVInteger *currentProgress = calloc(1, sizeof(CVInteger));
	CVInteger *shallStop = calloc(1, sizeof(CVInteger));
	*shallStop = 0;

	Py_BEGIN_ALLOW_THREADS
	CVParallelForStart(distributionsLoop, sentenceIndex, sentencesCount) {
		if (!*shallStop) {
			if (CVAtomicIncrementInteger(currentProgress) % updateInterval == 0) {
				CVParallelLoopCriticalRegionStart(distributionsLoop) {
					if (verbose) {
						printf("Walks: %" CVIndexScan "/%" CVIndexScan
							   " (%.2f%%)                                      "
							   "                "
							   "           \r",
							(CVIndex)(*currentProgress), sentencesCount,
							(*currentProgress) / (float)(sentencesCount - 1) * 100.0);
						fflush(stdout);
					}

					PyGILState_STATE gilState = PyGILState_Ensure();
					int signalResult = PyErr_CheckSignals();
					if (signalResult != 0) {
						*shallStop = 1;
						printf("Stopping Walks                                \n");
						fflush(stdout);
					} else if (callback) {
						PyObject *pArgs = Py_BuildValue(
							"nn", (Py_ssize_t)(*currentProgress), (Py_ssize_t)sentencesCount);
						if (pArgs) {
							PyObject *callbackResult = PyObject_Call(callback, pArgs, NULL);
							Py_DECREF(pArgs);
							if (!callbackResult) {
								*shallStop = 1;
							} else {
								Py_DECREF(callbackResult);
							}
						} else {
							*shallStop = 1;
						}
					}
					PyGILState_Release(gilState);
				}
				CVParallelLoopCriticalRegionEnd(distributionsLoop);
			}
		}
		if (!*shallStop) {
			CVIndex currentNode = nodesArray[sentenceIndex / walksPerNode];
			CVIndex previousNode = currentNode;
			CVUIntegerSet *previousNeighborsSet = CVNewUIntegerSet();
			unsigned int *seedRef = seeds + sentenceIndex;

			sentences[sentenceIndex * windowSize + 0] = currentNode + 1; // Always shifted by 1;
			if (q == 1.0 && p == 1.0) {
				for (CVIndex walkStep = 1; walkStep < windowSize; walkStep++) { //
					CVIndex *neighbors = network->vertexEdgesLists[currentNode];
					CVIndex neighborCount = network->vertexNumOfEdges[currentNode];
					CVIndex *neighEdges = network->vertexEdgesIndices[currentNode];
					if (neighborCount > 0) {
						CVFloat *probabilities = calloc(neighborCount, sizeof(CVFloat));
						for (CVIndex neighIndex = 0; neighIndex < neighborCount; neighIndex++) {
							CVIndex edgeIndex = neighEdges[neighIndex];
							CVFloat weight = 1.0;

							if (network->edgeWeighted) {
								weight = network->edgesWeights[edgeIndex];
							}
							probabilities[neighIndex] = weight;
						}
#ifdef __WIN32__
						unsigned int randomNumber;
						rand_s(&randomNumber);
						CVDouble choice = ((double)randomNumber / UINT_MAX);
#else
						CVDouble choice = ((double)rand_r(seedRef) / RAND_MAX);
#endif
						CVDistribution *distribution =
							CVCreateDistribution(probabilities, NULL, neighborCount);
						previousNode = currentNode;
						currentNode = neighbors[CVDistributionIndexForChoice(distribution, choice)];
						sentences[sentenceIndex * windowSize + walkStep] =
							currentNode + 1; // Always shifted by 1;
						CVDestroyDistribution(distribution);
						free(probabilities);
					} else {
						break;
					}
				}
			} else {
				for (CVIndex walkStep = 1; walkStep < windowSize; walkStep++) { //
					CVIndex *neighbors = network->vertexEdgesLists[currentNode];
					CVIndex neighborCount = network->vertexNumOfEdges[currentNode];
					CVIndex *neighEdges = network->vertexEdgesIndices[currentNode];
					if (neighborCount > 0) {
						CVFloat *probabilities = calloc(neighborCount, sizeof(CVFloat));
						for (CVIndex neighIndex = 0; neighIndex < neighborCount; neighIndex++) {
							CVIndex edgeIndex = neighEdges[neighIndex];
							CVIndex candidateIndex = neighbors[neighIndex];
							CVFloat weight = 1.0;

							if (network->edgeWeighted) {
								weight = network->edgesWeights[edgeIndex];
							}

							if (neighbors[neighIndex] == previousNode) {
								probabilities[neighIndex] = weight * 1 / p;
							} else if (CVUIntegerSetHas(previousNeighborsSet, candidateIndex)) {
								probabilities[neighIndex] = weight;
							} else {
								probabilities[neighIndex] = weight * 1 / q;
							}
						}

#ifdef __WIN32__
						unsigned int randomNumber;
						rand_s(&randomNumber);
						CVDouble choice = ((double)randomNumber / UINT_MAX);
#else
						CVDouble choice = ((double)rand_r(seedRef) / RAND_MAX);
#endif

						CVDistribution *distribution =
							CVCreateDistribution(probabilities, NULL, neighborCount);

						previousNode = currentNode;
						currentNode = neighbors[CVDistributionIndexForChoice(distribution, choice)];
						sentences[sentenceIndex * windowSize + walkStep] =
							currentNode + 1; // Always shifted by 1;
						CVDestroyDistribution(distribution);
						free(probabilities);

						CVUIntegerSetClear(previousNeighborsSet);
						for (CVIndex neighIndex = 0; neighIndex < neighborCount; neighIndex++) {
							CVUIntegerSetAdd(previousNeighborsSet, neighbors[neighIndex]);
						}
					} else {
						break;
					}
				}
			}
			CVUIntegerSetDestroy(previousNeighborsSet);
		}
	}
	CVParallelForEnd(distributionsLoop);
	Py_END_ALLOW_THREADS

	free(currentProgress);

	if (*shallStop) {
		printf("Stopped                                \n");
		// free(sentences);
		// free(shallStop);
		// PyErr_Format(PyExc_FileNotFoundError,"Error happened.");

		return NULL;
	}

	free(shallStop);

	if (verbose) {
		printf("DONE                                \n");
	}

	PyListObject *sentencesList = NULL;
	if (!outputFile) {
		sentencesList = PyList_New(sentencesCount);
	}
	for (CVIndex sentenceIndex = 0; sentenceIndex < sentencesCount; sentenceIndex++) {
		PyListObject *walkList = NULL;
		if (!outputFile) {
			walkList = PyList_New(0);
			PyList_SET_ITEM(sentencesList, sentenceIndex, walkList);
		}
		for (CVIndex walkStep = 0; walkStep < windowSize; walkStep++) {
			CVIndex nodeIndexWithOffset = sentences[sentenceIndex * windowSize + walkStep];
			if (nodeIndexWithOffset > 0) {
				if (outputFile) {
					if (labelsData) {
						fprintf(outputFile, "%s ", labelsData[(nodeIndexWithOffset - 1)]);
						// printf("\n%d:
						// %s\n",(int)(nodeIndexWithOffset-1),labelsData[nodeIndexWithOffset-1]);
					} else {
						fprintf(outputFile, "%" CVUIntegerScan " ", (nodeIndexWithOffset - 1));
					}
				} else {
					if (labelsData) {
						PyObject *value = Py_BuildValue("s", labelsData[(nodeIndexWithOffset - 1)]);
						// printf("\n%d:
						// %s\n",(int)(nodeIndexWithOffset-1),labelsData[nodeIndexWithOffset-1]);
						PyList_Append(walkList, value);
						Py_DECREF(value);
					} else {
						PyObject *value =
							PyLong_FromUnsignedLong((unsigned long)(nodeIndexWithOffset - 1));
						PyList_Append(walkList, value);
						Py_DECREF(value);
					}
				}
				// printf("%" CVUIntegerScan " ", (nodeIndexWithOffset - 1));
			} else {
				break;
			}
		}
		if (outputFile) {
			fprintf(outputFile, "\n");
		}
	}

	free(sentences);
	if (labelsData) {
		for (CVIndex index = 0; index < labelsDataSize; index++) {
			CVStringDestroy(labelsData[index]);
		}
		free(labelsData);
	}

	if (outputFile) {
		Py_RETURN_NONE;
	} else {
		return (PyObject *)sentencesList;
	}
}

PyObject *PyAgent_walkHits(PyAgent *self, PyObject *args, PyObject *kwds) {
	static char *kwlist[] = {"nodes", "p", "q", "windowSize", "walksPerNode", "batchSize",
		"verbose", "callback", "updateInterval", NULL};

	float p = 1.0;
	float q = 1.0;
	Py_ssize_t windowSize = 80;
	Py_ssize_t walksPerNode = 80;
	Py_ssize_t batchSize = 10000;
	int verbose = 0;
	PyObject *nodes = NULL;
	PyObject *callback = NULL;
	Py_ssize_t updateInterval = 1000;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OffnnnpOn", kwlist, &nodes, &p, &q, &windowSize,
			&walksPerNode, &batchSize, &verbose, &callback, &updateInterval)) {
		return NULL;
	}
	if (callback != NULL) {
		if (!PyCallable_Check(callback)) {
			PyErr_SetString(PyExc_ValueError, "Invalid callback.");
			printf("\n----ERROR----\nInvalid callback.\n");
			return NULL;
		}
	}
	windowSize += 1; // Our window size is in number of nodes, but users provide
					 // number of hops.
	CVNetwork *network = self->network;

	CVSize verticesCount = network->verticesCount;
	CVIndex *nodesArray = NULL;
	CVSize nodesArraySize = 0;
	// if nodes is not specified, put all nodes in the array
	if (!nodes) {
		nodesArray = calloc(verticesCount, sizeof(CVIndex));
		for (CVIndex index = 0; index < verticesCount; index++) {
			nodesArray[index] = index;
		}
		nodesArraySize = verticesCount;
	} else {
		PyArrayObject *nodesArrayObject = NULL;
		if (!(nodesArrayObject = convertToUIntegerArray(nodes, 1, 1))) {
			PyErr_SetString(PyExc_TypeError, "Error creating arrays.");
			printf("\n----ERROR----\nError creating arrays.\n");
			return NULL;
		}
		nodesArraySize = (CVSize)PyArray_SIZE(nodesArrayObject);
		nodesArray = PyArray_DATA(nodesArrayObject);
	}
	// check if indices are valid
	for (CVIndex index = 0; index < nodesArraySize; index++) {
		if (nodesArray[index] >= verticesCount) {
			free(nodesArray);
			PyErr_SetString(PyExc_TypeError, "Node indices should not be higher than the "
											 "number of vertices.");
			printf("\n----ERROR----\nNode indices should not be higher than "
				   "the number of vertices.\n");
			return NULL;
		}
	}

	// CVSize sentencesCount = nodesArraySize * walksPerNode;
	CVSize walksCount = nodesArraySize * walksPerNode;
	CVSize batchCount = (walksCount + batchSize - 1) / batchSize; // ceil(walksCount / batchSize);

	npy_intp dims[2] = {nodesArraySize, verticesCount};
	PyArrayObject *hitsArray = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_UINT64, 0);
	if (hitsArray == NULL) {
		PyErr_SetString(PyExc_TypeError, "Error creating arrays.");
		return NULL;
	}
	CVIndex *hitsMatrix = PyArray_DATA(hitsArray);
	unsigned int *seeds = calloc(batchCount, sizeof(unsigned int));

	unsigned int initialSeed = (unsigned int)time(NULL);
	for (CVIndex batchIndex = 0; batchIndex < batchCount; batchIndex++) {
#ifdef __WIN32__
		unsigned int randomNumber;
		rand_s(&randomNumber);
		randomNumber ^= (unsigned int)batchIndex;
		seeds[batchIndex] = randomNumber;
#else
		seeds[batchIndex] = rand_r(&initialSeed) ^ (unsigned int)batchIndex;
#endif
	}

	CVInteger *currentProgress = calloc(1, sizeof(CVInteger));
	CVInteger *shallStop = calloc(1, sizeof(CVInteger));
	*shallStop = 0;
	Py_BEGIN_ALLOW_THREADS
	CVParallelForStart(distributionsLoop, batchIndex, batchCount) {
		if (CVUnlikely(!*shallStop)) {
			if (CVUnlikely(CVAtomicIncrementInteger(currentProgress) % updateInterval == 0)) {
				CVParallelLoopCriticalRegionStart(distributionsLoop) {
					if (verbose) {
						printf("Batches: %" CVIndexScan "/%" CVIndexScan
							   " (%.2f%%)                                      "
							   "                "
							   "           \r",
							(CVIndex)(*currentProgress), batchCount,
							(*currentProgress) / (float)(batchCount) * 100.0);
						fflush(stdout);
					}

					PyGILState_STATE gilState = PyGILState_Ensure();
					int signalResult = PyErr_CheckSignals();
					if (signalResult != 0) {
						*shallStop = 1;
						printf("Stopping Walks                                \n");
						fflush(stdout);
					} else if (callback) {
						PyObject *pArgs = Py_BuildValue(
							"nn", (Py_ssize_t)(*currentProgress), (Py_ssize_t)batchCount);
						if (pArgs) {
							PyObject *callbackResult = PyObject_Call(callback, pArgs, NULL);
							Py_DECREF(pArgs);
							if (!callbackResult) {
								*shallStop = 1;
							} else {
								Py_DECREF(callbackResult);
							}
						} else {
							*shallStop = 1;
						}
					}
					PyGILState_Release(gilState);
				}
				CVParallelLoopCriticalRegionEnd(distributionsLoop);
			}
		}
		unsigned int *seedRef = seeds + batchIndex;
		CVIndex walkStartIndex = batchIndex * batchSize;
		CVIndex walkEndIndex = CVMIN((batchIndex + 1) * batchSize, nodesArraySize * walksPerNode);

		for (CVIndex walkIndex = walkStartIndex; walkIndex < walkEndIndex; walkIndex++) {
			if (CVLikely(!*shallStop)) {
				CVIndex sourceNodeIndex = walkIndex / walksPerNode;
				CVIndex sourceNode = nodesArray[walkIndex / walksPerNode];
				CVIndex currentNode = sourceNode;
				CVIndex previousNode = currentNode;

				// sentences[sentenceIndex * windowSize + 0] =
				// 	currentNode + 1; // Always shifted by 1;

				// fastest way to check if p and q are both close to 1
				// if (CVLikely((fabs(p - 1.0) < 1e-6 && fabs(q - 1.0) < 1e-6)))
				// {
				if (CVLikely(p == 1.0 && q == 1.0)) {
					if (CVLikely(!network->edgeWeighted)) {
						for (CVIndex walkStep = 1; walkStep < windowSize; walkStep++) { //
							CVIndex *neighbors = network->vertexEdgesLists[currentNode];
							CVIndex neighborCount = network->vertexNumOfEdges[currentNode];
							if (CVLikely(neighborCount > 0)) {
								CVDouble choice = getRandomChoice(seedRef);
								previousNode = currentNode;
								currentNode = neighbors[(CVIndex)(choice * neighborCount)];
								// sentences[sentenceIndex * windowSize +
								// walkStep] = 	currentNode + 1; // Always
								// shifted by 1;
								CVAtomicIncrementInteger(
									hitsMatrix + (sourceNodeIndex * verticesCount + currentNode));
							} else {
								break;
							}
						}
					} else {
						for (CVIndex walkStep = 1; walkStep < windowSize; walkStep++) { //
							CVIndex *neighbors = network->vertexEdgesLists[currentNode];
							CVIndex neighborCount = network->vertexNumOfEdges[currentNode];
							if (neighborCount > 0) {
								CVDouble choice = getRandomChoice(seedRef);
								previousNode = currentNode;
								CVIndex *neighEdges = network->vertexEdgesIndices[currentNode];
								CVFloat *probabilities = calloc(neighborCount, sizeof(CVFloat));
								for (CVIndex neighIndex = 0; neighIndex < neighborCount;
									 neighIndex++) {
									CVIndex edgeIndex = neighEdges[neighIndex];
									CVFloat weight = network->edgesWeights[edgeIndex];
									probabilities[neighIndex] = weight;
								}
								CVDistribution *distribution =
									CVCreateDistribution(probabilities, NULL, neighborCount);
								currentNode =
									neighbors[CVDistributionIndexForChoice(distribution, choice)];
								CVDestroyDistribution(distribution);
								free(probabilities);
								// sentences[sentenceIndex * windowSize +
								// walkStep] = 	currentNode + 1; // Always
								// shifted by 1;
								CVAtomicIncrementInteger(
									hitsMatrix + (sourceNodeIndex * verticesCount + currentNode));
							} else {
								break;
							}
						}
					}
				} else {
					CVUIntegerSet *previousNeighborsSet = CVNewUIntegerSet();
					for (CVIndex walkStep = 1; walkStep < windowSize; walkStep++) { //
						CVIndex *neighbors = network->vertexEdgesLists[currentNode];
						CVIndex neighborCount = network->vertexNumOfEdges[currentNode];
						if (neighborCount > 0) {
							CVFloat *probabilities = calloc(neighborCount, sizeof(CVFloat));
							CVIndex *neighEdges = network->vertexEdgesIndices[currentNode];
							for (CVIndex neighIndex = 0; neighIndex < neighborCount; neighIndex++) {
								CVIndex edgeIndex = neighEdges[neighIndex];
								CVIndex candidateIndex = neighbors[neighIndex];
								CVFloat weight = 1.0;

								if (network->edgeWeighted) {
									weight = network->edgesWeights[edgeIndex];
								}

								if (neighbors[neighIndex] == previousNode) {
									probabilities[neighIndex] = weight * 1 / p;
								} else if (CVUIntegerSetHas(previousNeighborsSet, candidateIndex)) {
									probabilities[neighIndex] = weight;
								} else {
									probabilities[neighIndex] = weight * 1 / q;
								}
							}

							CVDouble choice = getRandomChoice(seedRef);

							CVDistribution *distribution =
								CVCreateDistribution(probabilities, NULL, neighborCount);

							previousNode = currentNode;
							currentNode =
								neighbors[CVDistributionIndexForChoice(distribution, choice)];

							// sentences[sentenceIndex * windowSize + walkStep]
							// = 	currentNode + 1; // Always shifted by 1;
							CVAtomicIncrementInteger(
								hitsMatrix + (sourceNodeIndex * verticesCount + currentNode));

							CVDestroyDistribution(distribution);
							free(probabilities);

							CVUIntegerSetClear(previousNeighborsSet);
							for (CVIndex neighIndex = 0; neighIndex < neighborCount; neighIndex++) {
								CVUIntegerSetAdd(previousNeighborsSet, neighbors[neighIndex]);
							}
						} else {
							break;
						}
					}
					CVUIntegerSetDestroy(previousNeighborsSet);
				}
			} else {
				break;
			}
		}
	}
	CVParallelForEnd(distributionsLoop);
	Py_END_ALLOW_THREADS

	free(currentProgress);

	if (*shallStop) {
		printf("Stopped                                \n");
		// free(sentences);
		free(shallStop);
		// PyErr_Format(PyExc_FileNotFoundError,"Error happened.");

		return NULL;
	}

	free(shallStop);

	if (verbose) {
		printf("DONE                                \n");
	}
	return (PyObject *)hitsArray;
}






PyObject *PyAgent_centrality(PyAgent *self, PyObject *args, PyObject *kwds) {
	static char *kwlist[] = {"weighted", NULL};

	int verbose = 0;
	PyObject *useWeights = NULL;

	if (!PyArg_ParseTupleAndKeywords(args, kwds, "|O", kwlist, &useWeights)) {
		return NULL;
	}
	CVBool weighted = CVFalse;
	if (useWeights) {
		if (PyObject_IsTrue(useWeights)) {
			weighted = CVTrue;
		}
	}

	CVNetwork *network = self->network;
	CVSize verticesCount = network->verticesCount;
	CVDoubleArray centralityDoubleArray;
	CVDoubleArrayInitWithCapacity(verticesCount, &centralityDoubleArray);
	npy_intp dims[1] = {verticesCount};
	PyArrayObject *centralityArray = (PyArrayObject *)PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
	if (centralityArray == NULL) {
		PyErr_SetString(PyExc_TypeError, "Error creating arrays.");
		return NULL;
	}
	CVDouble *centralityBuffer = PyArray_DATA(centralityArray);
	CVNetworkCalculateCentrality(network, &centralityDoubleArray, NULL);
	for (CVIndex index = 0; index < verticesCount; index++) {
		centralityBuffer[index] = centralityDoubleArray.data[index]/2.0;
	}
	CVDoubleArrayDestroy(&centralityDoubleArray);
	return (PyObject *)centralityArray;
}

static PyMethodDef PyAgent_methods[] = {
	{"generateWalks", (PyCFunction)PyAgent_generateWalks, METH_VARARGS | METH_KEYWORDS,
		"Create a sequence of walks."},
	{"walkHits", (PyCFunction)PyAgent_walkHits, METH_VARARGS | METH_KEYWORDS,
		"Calculate walk hits from source node."},
	{"betweenness", (PyCFunction)PyAgent_centrality, METH_VARARGS | METH_KEYWORDS,
		"Calculate centrality or weighted centrality for the network."},
	{NULL} /* Sentinel */
};

static PyTypeObject PyAgentType = {
	PyVarObject_HEAD_INIT(NULL, 0).tp_name = "cxrandomwalk_core.Agent",
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

char cxrandomwalk_coremod_docs[] = "This is CXRandomWalk module.";

static PyModuleDef cxrandomwalk_core_mod = {PyModuleDef_HEAD_INIT, .m_name = "cxrandomwalk_core",
	.m_doc = cxrandomwalk_coremod_docs, .m_size = -1, .m_methods = NULL, .m_slots = NULL,
	.m_traverse = NULL, .m_clear = NULL, .m_free = NULL};

PyMODINIT_FUNC PyInit_cxrandomwalk_core(void) {
	import_array();

	PyObject *m;
	if (PyType_Ready(&PyAgentType) < 0) {
		return NULL;
	}
	m = PyModule_Create(&cxrandomwalk_core_mod);
	if (m == NULL) {
		return NULL;
	}
	Py_INCREF(&PyAgentType);
	if (PyModule_AddObject(m, "Agent", (PyObject *)&PyAgentType) < 0) {
		Py_DECREF(&PyAgentType);
		Py_DECREF(m);
		return NULL;
	}

	if (PyModule_AddStringConstant(m, "__version__", CVTOKENTOSTRING(k_PYCXVersion)) < 0) {
		Py_DECREF(m);
		return NULL;
	}

	return m;
}
