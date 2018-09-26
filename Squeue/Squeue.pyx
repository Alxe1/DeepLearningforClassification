#-*-coding:utf-8-*-
#queue.pyx
#distutils: sources = c-algorithms/src/queue.c
#distutils: include_dirs = c-algorithms/src/
cimport cqueue

cdef class Squeue:
	cdef cqueue.Queue *_c_queue
	
	#初始化
	def __cinit__(self):
		self._c_queue = cqueue.queue_new()
		if self._c_queue is NULL:
			raise MemoryError()
	
	#释放内存
	def __dealloc__(self):
		if self._c_queue is not NULL:
			cqueue.queue_free(self._c_queue)
			
	cpdef append(self, int value):
		#append right
		if not cqueue.queue_push_tail(self._c_queue, <void*>value):
			raise MemoryError()
			
	cpdef append_left(self, int value):
		#append left
		if not cqueue.queue_push_head(self._c_queue, <void*>value):
			raise MemoryError()
			
	cpdef extend(self, values):
		for value in values:
			self.append(value)
			
	cdef extend_ints(self, int* values, size_t count):
		cdef int value
		for value in values[:count]: # Slicing pointer to limit the iteration boundaries.
			self.append(value)
			
	cpdef int peek(self) except? -1:
		#the first element of queue
		cdef int value = <Py_ssize_t>cqueue.queue_peek_head(self._c_queue)
		if value == 0:
			#this may mean that the queue is empty, or that it happends to contain a 0 value
			if cqueue.queue_is_empty(self._c_queue):
				raise IndexError("Queue is empty")
		return value
		
	cpdef int peek_tail(self) except? -1:
		#the last element of queue
		cdef int value = <Py_ssize_t>cqueue.queue_peek_tail(self._c_queue)
		if value == 0:
			#this may mean that the queue is empty, or that it happends to contain a 0 value
			if cqueue.queue_is_empty(self._c_queue):
				raise IndexError("Queue is empty")
		return value
	
	cpdef int pop_left(self) except? -1:
		#pop left
		if cqueue.queue_is_empty(self._c_queue):
			raise IndexError("Queue is empty")
		return <Py_ssize_t>cqueue.queue_pop_head(self._c_queue)
		
	cpdef int pop(self) except? -1:
		#pop right
		if cqueue.queue_is_empty(self._c_queue):
			raise IndexError("Queue is empty")
		return <Py_ssize_t>cqueue.queue_pop_tail(self._c_queue)
		
	def __bool__(self):
		return not cqueue.queue_is_empty(self._c_queue)
		
		