#pragma once

#if defined( __CUDACC__ ) || defined( __HIPCC__ )
#define TYPED_BUFFER_DEVICE_FUNC __device__
#else
#include <Orochi/Orochi.h>
#define TYPED_BUFFER_DEVICE_FUNC
#endif

enum TYPED_BUFFER_TYPE
{
	TYPED_BUFFER_HOST = 0,
	TYPED_BUFFER_DEVICE = 1,
};

template<class T>
struct TypedBuffer
{
	T* m_data = nullptr;
	size_t m_size : 63;
	size_t m_isDevice : 1;

	TYPED_BUFFER_DEVICE_FUNC
	TypedBuffer(const TypedBuffer&) = delete;

	TYPED_BUFFER_DEVICE_FUNC
	void operator=(const TypedBuffer&) = delete;

#if defined( __CUDACC__ ) || defined( __HIPCC__ )
	TYPED_BUFFER_DEVICE_FUNC
	TypedBuffer() :m_size(0), m_isDevice(TYPED_BUFFER_DEVICE)
	{
	}
#else
	void allocate(size_t n)
	{
		if (m_isDevice)
		{
			if (m_data)
			{
				oroFree(m_data);
			}
			oroMalloc((void**)&m_data, n * sizeof(T));
		}
		else
		{
			if (m_data)
			{
				free(m_data);
			}
			m_data = (T*)malloc(n * sizeof(T));
		}
		m_size = n;
	}

	TypedBuffer(TYPED_BUFFER_TYPE type) :m_size(0), m_isDevice(type)
	{
	}

	~TypedBuffer()
	{
		if (m_data)
		{
			if (m_isDevice)
			{
				oroFree(m_data);
			}
			else
			{
				free(m_data);
			}
		}
	}

	TypedBuffer(TypedBuffer<T>&& other) 
		:m_data(other.m_data),
		m_size(other.m_size),
		m_isDevice(other.m_isDevice) 
	{
		other.m_data = nullptr;
		other.m_size = 0;
	}

	TypedBuffer<T> toHost() const
	{
		TypedBuffer<T> r(TYPED_BUFFER_HOST);
		r.allocate(size());
		oroMemcpyDtoH(r.data(), m_data, m_size * sizeof(T));
		return r;
	}
	TypedBuffer<T> toDevice() const
	{
		TypedBuffer<T> r(TYPED_BUFFER_DEVICE);
		r.allocate(size());
		oroMemcpyHtoD(r.data(), m_data, m_size * sizeof(T));
		return r;
	}
#endif

	TYPED_BUFFER_DEVICE_FUNC
	size_t size() const { return m_size; }

	TYPED_BUFFER_DEVICE_FUNC
	size_t bytes() const { return m_size * sizeof(T); }

	TYPED_BUFFER_DEVICE_FUNC
	const T* data() const { return m_data; }

	TYPED_BUFFER_DEVICE_FUNC
	T* data() { return m_data; }

	TYPED_BUFFER_DEVICE_FUNC
	const T* begin() const { return data(); }

	TYPED_BUFFER_DEVICE_FUNC
	const T* end() const { return data() + m_size; }

	TYPED_BUFFER_DEVICE_FUNC
	T* begin() { return data(); }

	TYPED_BUFFER_DEVICE_FUNC
	T* end() { return data() + m_size; }

	TYPED_BUFFER_DEVICE_FUNC
	const T& operator[](int index) const { return m_data[index]; }

	TYPED_BUFFER_DEVICE_FUNC
	T& operator[](int index) { return m_data[index]; }

	TYPED_BUFFER_DEVICE_FUNC
	bool isDevice() const { return m_isDevice; }

	TYPED_BUFFER_DEVICE_FUNC
	bool isHost() const { return !isDevice(); }
};