#include	<cuda_runtime.h>
#include	<cublas_v2.h>
#include	<sys/mman.h>
#include	<sys/stat.h>
#include	<sys/time.h>
#include	<algorithm>
#include	<iostream>
#include	<stdint.h>
#include	<unistd.h>
#include	<cstdlib>
#include	<fcntl.h>
#include	<vector>
#include	<cfloat>
using	namespace	std;
#define	tiger_beta	0.03125f
cublasHandle_t	handle;
float	eta;
uint64_t	prng=time(NULL);
static inline uint64_t wyrand(uint64_t	*seed){	*seed+=0xa0761d6478bd642full;	uint64_t  see1=*seed^0xe7037ed1a0b428dbull;	see1*=(see1>>32)|(see1<<32);	return	(*seed*((*seed>>32)|(*seed<<32)))^((see1>>32)|(see1<<32));	}
static inline float wy2gau(uint64_t r){	const float	_wynorm=1.0/(1ull<<20);	return ((r&0x1fffff)+((r>>21)&0x1fffff)+((r>>42)&0x1fffff))*_wynorm-3.0f;	}
static inline double wy2u01(uint64_t r){	const double _wynorm=1.0/(1ull<<52);	return (r>>12)*_wynorm;	}
__device__ void _wymum(uint64_t *A,	uint64_t *B){	uint64_t	hh=(*A>>32)*(*B>>32), hl=(*A>>32)*(uint32_t)*B, lh=(uint32_t)*A*(*B>>32), ll=(uint64_t)(uint32_t)*A*(uint32_t)*B;	*A=((hl>>32)|(hl<<32))^hh;	*B=((lh>>32)|(lh<<32))^ll;	}
__device__	uint64_t	_wyhash64(uint64_t	A,	uint64_t	B){	A^=0xa0761d6478bd642full;	B^=0xa0761d6478bd642full;	_wymum(&A,&B);	A^=0xa0761d6478bd642full;	B^=0xa0761d6478bd642full;	_wymum(&A,&B);	return	A^B;	}
__global__	void	_tiger(float	*w, float	*m,	float	lr){	
	unsigned	i=(blockIdx.x*blockDim.x+threadIdx.x)<<2;
	float4	*w4=(float4*)(w+i),*m4=(float4*)(m+i);
	*w4=make_float4(w4->x-lr*((m4->x>0)-0.5f),w4->y-lr*((m4->y>0)-0.5f),w4->z-lr*((m4->z>0)-0.5f),w4->w-lr*((m4->w>0)-0.5f));
}
struct	bfloat8{	__nv_bfloat162	x,y,z,w;	};
struct	float8x{	float2	x,y,z,w;	};
__global__	void	_quant(float	*inp, __nv_bfloat16	*out){	
	unsigned	i=(blockIdx.x*blockDim.x+threadIdx.x)<<3;
	float8x	*i4=(float8x*)(inp+i);	bfloat8	o8;
	o8.x=__float22bfloat162_rn(i4->x);	o8.y=__float22bfloat162_rn(i4->y);	o8.z=__float22bfloat162_rn(i4->z);	o8.w=__float22bfloat162_rn(i4->w);
	*(bfloat8*)(out+i)=o8;
}
template<unsigned	N>
struct	Data16{
	__nv_bfloat16	*data;
	Data16(){	cudaMallocManaged(&data,	N*sizeof(__nv_bfloat16));	}
	~Data16(){	cudaFree(data);	}
};
__global__	void	_s16(unsigned	N,	float	*w, __nv_bfloat16	*g){	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x;	if(id<N)	g[id]=__float2bfloat16(w[id]);	}
__global__	void	_l16(unsigned	N,	float	*w, __nv_bfloat16	*g){	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x;	if(id<N)	w[id]=__bfloat162float(g[id]);	}
template<unsigned	N>
struct	Data{
	float	*data;
	Data(){	cudaMallocManaged(&data,	N*sizeof(float));	}
	~Data(){	cudaFree(data);	}
	void	save(FILE	*F){
		Data16<N>	tmp;	
		_s16<<<(N+15)/16,16>>>(N,data,tmp.data);	
		cudaDeviceSynchronize();	
		fwrite(tmp.data,N*2,1,F);	
	}
	void	load(FILE	*F){
		Data16<N>	tmp;	
		if(fread(tmp.data,N*2,1,F)!=1){	return;	}	
		_l16<<<(N+15)/16,16>>>(N,data,tmp.data);	
		cudaDeviceSynchronize();	
	}
	unsigned	size(void){	return	N;	}
	void	zero(void){	cudaMemset(data,	0,	N*sizeof(float));	}
	void	rand(float	norm=1){	for(unsigned	i=0;	i<N;	i++)	data[i]=norm*wy2gau(wyrand(&prng));	}
	float	norm(void){	float	n;	cublasSnrm2(handle,N,data,1,&n);	cudaDeviceSynchronize();	return	n/sqrtf(N);	}
};
template<unsigned	R0,	unsigned	R1,	unsigned	C>
struct	linear{
	Data16<R0*R1>	weq;
	Data<R0*R1>	wei,wem;
	Data16<R0*C>	inq;
	Data16<R1*C>	giq;
	Data<R1*C>	out;
	linear(){	wei.rand(1/sqrtf(R0));	wem.zero();	}
	void	save(FILE	*F){	wei.save(F);	}
	void	load(FILE	*F){	wei.load(F);	}
	unsigned	size(void){	return	wei.size();	}
	uint64_t	flop(void){	return	6ull*R1*C*R0;	}
	void	fw(Data<R0*C>	&inp){
		float	alf=1/sqrtf(R0),	bet=0;
		_quant<<<R0*R1/8/16,16>>>(wei.data,weq.data);	_quant<<<R0*C/8/16,16>>>(inp.data,inq.data);
		cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,R1,C,R0,&alf,weq.data,CUDA_R_16BF,R0,inq.data,CUDA_R_16BF,R0,&bet,out.data,CUDA_R_32F,R1,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
	}
	void	bk(Data<R0*C>	&inp,	Data<R1*C>	&gin,	Data<R0*C>	&gra,	bool	accumulate=false){
		float	alf=1/sqrtf(R0),	alf1=tiger_beta/sqrtf(R0*C),	bet=1-tiger_beta,	bet1=accumulate;
		_quant<<<R1*C/8/16,16>>>(gin.data,giq.data);
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,R0,R1,C,&alf1,inq.data,CUDA_R_16BF,R0,giq.data,CUDA_R_16BF,R1,&bet,wem.data,CUDA_R_32F,R0,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);		
		cublasGemmEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,R0,C,R1,&alf,weq.data,CUDA_R_16BF,R0,giq.data,CUDA_R_16BF,R1,&bet1,gra.data,CUDA_R_32F,R0,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
		_tiger<<<R0*R1/4/16,16>>>(wei.data,wem.data,eta);
	}
};
__global__	void	_layernormf(unsigned	R,	unsigned	C,	unsigned	H,	float	*inp,	float	*norm){
	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x,	r=R/H;
	float	*in=inp+id*r,	sum=0,	nor=0;
	for(unsigned	i=0;	i<r;	i+=4){	float4*	t=(float4*)(in+i);	sum+=t->x+t->y+t->z+t->w;	nor+=t->x*t->x+t->y*t->y+t->z*t->z+t->w*t->w;	}
	sum/=r;	nor=fmaxf(nor-sum*sum*r,1e-18f);	norm[id]=nor;	nor=sqrtf(r/nor);
	for(unsigned	i=0;	i<r;	i+=4){	float4	*t=(float4*)(in+i);	*t=make_float4((t->x-sum)*nor,(t->y-sum)*nor,(t->z-sum)*nor,(t->w-sum)*nor);	}
}
__global__	void	_layernormb(unsigned	R,	unsigned	C,	unsigned	H,	float	*inp,	float	*gin,	float	*norm){
	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x,	r=R/H;
	float	*gi=gin+id*r,	*ou=inp+id*r,	mg=0,	 sgi=0,sou=0,	s=sqrtf(r/norm[id]),	sum=0;
	for(unsigned	i=0;	i<r;	i+=4){	float4	*g=(float4*)(gi+i),	*o=(float4*)(ou+i);	sgi+=g->x+g->y+g->z+g->w;	sou+=o->x+o->y+o->z+o->w;	mg+=g->x*o->x+g->y*o->y+g->z*o->z+g->w*o->w;	}
	mg/=norm[id]*s;	sum=(s*sgi-mg*sou)/r;
	for(unsigned	i=0;	i<r;	i+=4){	float4	*g=(float4*)(gi+i),	*o=(float4*)(ou+i);	*g=make_float4(s*g->x-mg*o->x-sum,s*g->y-mg*o->y-sum,s*g->z-mg*o->z-sum,s*g->w-mg*o->w-sum);	}
}
template<unsigned	R,	unsigned	C,	unsigned	H>
struct	layernorm{
	Data<C*H>	nor;
	void	fw(Data<R*C>	&inp){	_layernormf<<<C*H/16,16>>>(R,C,H,inp.data,nor.data);	}
	void	bk(Data<R*C>	&inp,	Data<R*C>	&gin){	_layernormb<<<C*H/16,16>>>(R,C,H,inp.data,gin.data,nor.data);	}
};
__global__	void	_softmaxf(unsigned	R,	float	*inp){
	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x;
	float	*p=inp+id*R,	sum=0,	ma=-FLT_MAX;
	for(unsigned	i=0;    i<R;    i++)	ma=fmaxf(p[i],ma);
	for(unsigned	i=0;	i<R;	i++)	sum+=(p[i]=expf(p[i]-ma));
	for(unsigned	i=0;	i<R;	i++)	p[i]/=sum;
}
__global__	void	_sexyfp(unsigned	C,	float	*att,	float	*pe){
	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x,	c=id%C,	h=id/C;
	float	*a=att+id*C,	*p=pe+h*C+c;
	for(unsigned	i=0;	i<=c;	i++)	a[i]=expf(*(p-i)+a[i]);
	for(unsigned	i=c+1;	i<C;	i++)	a[i]=0;
}
__global__	void	_sexyfsuv(unsigned	R,	float	*u,	float	*v,	float	*out){
	unsigned	id=(blockIdx.x*blockDim.x+threadIdx.x)<<2;
	float4	*u4=(float4*)(u+(id/R)*4*R+3*R+(id%R)),	*v4=(float4*)(v+id),	*o4=(float4*)(out+id);
	*o4=make_float4(u4->x*v4->x,u4->y*v4->y,u4->z*v4->z,u4->w*v4->w);
}
__global__	void	_sexybsuv(unsigned	R,	float	*u,	float	*v,	float	*gin,	float	*gx){
	unsigned	id=(blockIdx.x*blockDim.x+threadIdx.x)<<2;
	float4	*u4=(float4*)(u+(id/R)*4*R+3*R+(id%R)),	*v4=(float4*)(v+id),	*g4=(float4*)(gin+id),	*x4=(float4*)(gx+(id/R)*4*R+3*R+(id%R));
	*x4=make_float4(v4->x*g4->x,v4->y*g4->y,v4->z*g4->z,v4->w*g4->w);
	*v4=make_float4(u4->x*g4->x,u4->y*g4->y,u4->z*g4->z,u4->w*g4->w);
}
__global__	void	_sexyba(float	*gin,	float	*att){
	unsigned	id=(blockIdx.x*blockDim.x+threadIdx.x)<<2;
	float4	*g4=(float4*)(gin+id),	*a4=(float4*)(att+id);
	*g4=make_float4(a4->x*g4->x,a4->y*g4->y,a4->z*g4->z,a4->w*g4->w);
}
__global__	void	_sexybp(unsigned	R,	unsigned	C,	float	*a,	float	*pe,	float	*pm,	float	eta){
	unsigned  id=blockIdx.x*blockDim.x+threadIdx.x,	h=id/C,	c=id%C;	float	s=0,*p=a+h*C*C;
	for(unsigned	i=c;	i<C;	i++)	s+=p[i*C+(i-c)];
	s/=sqrtf((C-c)*R);	pm[id]+=tiger_beta*(s-pm[id]);	pe[id]-=eta*((pm[id]>0)-0.5f);
}
__global__	void	_sexyadd(unsigned	R,	unsigned	H,	float	*inp,	float	*out){
	unsigned  id=(blockIdx.x*blockDim.x+threadIdx.x)<<2,c=id/R,r=id%R;	float4	s={},*o4=(float4*)(out+id);
	for(unsigned	h=0;	h<H;	h++){	float4*	i4=(float4*)(inp+c*R*H+h*R+r);	s=make_float4(s.x+i4->x,s.y+i4->y,s.z+i4->z,s.w+i4->w);	}
	*o4=make_float4(s.x+o4->x,s.y+o4->y,s.z+o4->z,s.w+o4->w);
}
template<unsigned	R,	unsigned	C,	unsigned	H>
struct	sexy{
	static	Data<C*C*H>	da;
	static	Data<R*C>	gi;
	static	Data<4*R*C>	gx;
	static	Data16<R*C>	vaq;
	Data16<4*R*C>	xq;
	Data16<C*C*H>	atq;
	Data<C*C*H>	at;
	Data<R*C>	va,tmp;
	layernorm<R,C,H>	n1;
	layernorm<4*R,C,4*H>	n4;
	Data<H*C>	pe,pm;
	linear<R,4*R,C>	x;
	linear<R,R,C>	o;
	Data<R*C>	&out=o.out;
	sexy(){	pe.zero();	pm.zero();	}
	void	save(FILE	*F){	pe.save(F);	x.save(F);	o.save(F);	}
	void	load(FILE	*F){	pe.load(F);	x.load(F);	o.load(F);	}
	unsigned	size(void){	return	pe.size()+x.size()+o.size();	}
	uint64_t	flop(void){	return	x.flop()+o.flop()+12ull*C*C*R;	}
	void	fw(Data<R*C>	&inp){
		float	alf=1/sqrtf(R/H),alf1=1,bet=0;
		x.fw(inp);	n4.fw(x.out);	_quant<<<4*R*C/8/16,16>>>(x.out.data,xq.data);
		cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,C,C,R/H,&alf,xq.data,CUDA_R_16BF,4*R,R/H,xq.data+R,CUDA_R_16BF,4*R,R/H,&bet,at.data,CUDA_R_32F,C,C*C,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
		_sexyfp<<<C*H/16,16>>>(C,at.data,pe.data);	_quant<<<C*C*H/8/16,16>>>(at.data,atq.data);
		cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,R/H,C,C,&alf1,xq.data+2*R,CUDA_R_16BF,4*R,R/H,atq.data,CUDA_R_16BF,C,C*C,&bet,va.data,CUDA_R_32F,R,R/H,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
		_sexyfsuv<<<R*C/4/16,16>>>(R,x.out.data,va.data,tmp.data);	n1.fw(tmp);	o.fw(tmp);
		cublasSaxpy(handle,R*C,&alf1,inp.data,1,out.data,1);
	}
	void	bk(Data<R*C>	&inp,	Data<R*C>	&gin,	Data<R*C> &gra){
		float	alf=1/sqrtf(R/H),alf1=1,bet=0;
		o.bk(tmp,gin,gi);	n1.bk(tmp,gi);	_sexybsuv<<<R*C/4/16,16>>>(R,x.out.data,va.data,gi.data,gx.data);	_quant<<<R*C/8/16,16>>>(va.data,vaq.data);
		cublasGemmStridedBatchedEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,C,C,R/H,&alf1,xq.data+2*R,CUDA_R_16BF,4*R,R/H,vaq.data,CUDA_R_16BF,R,R/H,&bet,da.data,CUDA_R_32F,C,C*C,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
		cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,R/H,C,C,&alf1,vaq.data,CUDA_R_16BF,R,R/H,atq.data,CUDA_R_16BF,C,C*C,&bet,gx.data+2*R,CUDA_R_32F,4*R,R/H,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
		_sexyba<<<C*C*H/4/16,16>>>(da.data,at.data);	_sexybp<<<H*C/16,16>>>(R,C,da.data,pe.data,pm.data,eta);	_quant<<<C*C*H/8/16,16>>>(da.data,atq.data);
		cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_T,R/H,C,C,&alf,xq.data+R,CUDA_R_16BF,4*R,R/H,atq.data,CUDA_R_16BF,C,C*C,&bet,gx.data,CUDA_R_32F,4*R,R/H,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
		cublasGemmStridedBatchedEx(handle,CUBLAS_OP_N,CUBLAS_OP_N,R/H,C,C,&alf,xq.data,CUDA_R_16BF,4*R,R/H,atq.data,CUDA_R_16BF,C,C*C,&bet,gx.data+R,CUDA_R_32F,4*R,R/H,H,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
		n4.bk(x.out,gx);	x.bk(inp,gx,gra);	cublasSaxpy(handle,R*C,&alf1,gin.data,1,gra.data,1);
	}
};
template<unsigned	R,	unsigned	C,	unsigned	H>
Data<C*C*H>	sexy<R,C,H>::da;
template<unsigned	R,	unsigned	C,	unsigned	H>
Data<R*C>	sexy<R,C,H>::gi;
template<unsigned	R,	unsigned	C,	unsigned	H>
Data16<R*C>	sexy<R,C,H>::vaq;
template<unsigned	R,	unsigned	C,	unsigned	H>
Data<4*R*C>	sexy<R,C,H>::gx;
__global__	void	_selffsuv(unsigned	S,	float	*u,	float	*out){
	unsigned	id=(blockIdx.x*blockDim.x+threadIdx.x)<<2;
	float4	*u4=(float4*)(u+(id/S)*2*S+(id%S)),	*v4=(float4*)(u+(id/S)*2*S+S+(id%S)),	*o4=(float4*)(out+id);
	*o4=make_float4(u4->x*v4->x,u4->y*v4->y,u4->z*v4->z,u4->w*v4->w);
}
__global__	void	_selfbsuv(unsigned	S,	float	*u,	float	*gin,	float	*d){
	unsigned	id=(blockIdx.x*blockDim.x+threadIdx.x)<<2;
	float4  *u4=(float4*)(u+(id/S)*2*S+(id%S)),	*v4=(float4*)(u+(id/S)*2*S+S+(id%S)),	*g4=(float4*)(gin+id),	*p4=(float4*)(d+(id/S)*2*S+(id%S)),	*q4=(float4*)(d+(id/S)*2*S+S+(id%S));
	*p4=make_float4(v4->x*g4->x,v4->y*g4->y,v4->z*g4->z,v4->w*g4->w);	*q4=make_float4(u4->x*g4->x,u4->y*g4->y,u4->z*g4->z,u4->w*g4->w);
}
template<unsigned	R,	unsigned	C,	unsigned	H>
struct	self{
	static	Data<R*C>	gi;
	static	Data<2*R*C>	du;
	Data<R*C>	tmp;
	layernorm<2*R,C,2*H>	n2;
	layernorm<R,C,H>	n1;
	linear<R,2*R,C>	u;
	linear<R,R,C>	o;
	Data<R*C>	&out=o.out;
	void	save(FILE	*F){	u.save(F);	o.save(F);	}
	void	load(FILE	*F){	u.load(F);	o.load(F);	}
	unsigned	size(void){	return	u.size()+o.size();	}
	uint64_t	flop(void){	return	u.flop()+o.flop();	}
	void	fw(Data<R*C>	&inp){
		float	alf1=1;
		u.fw(inp);	n2.fw(u.out);
		_selffsuv<<<R*C/4/16,16>>>(R,u.out.data,tmp.data);	n1.fw(tmp);	o.fw(tmp);
		cublasSaxpy(handle,R*C,&alf1,inp.data,1,out.data,1);
	}
	void	bk(Data<R*C>	&inp,	Data<R*C>	&gin,	Data<R*C> &gra){
		float	alf1=1;
		o.bk(tmp,gin,gi);	n1.bk(tmp,gi);
		_selfbsuv<<<R*C/4/16,16>>>(R,u.out.data,gi.data,du.data);	
		n2.bk(u.out,du);	u.bk(inp,du,gra);
		cublasSaxpy(handle,R*C,&alf1,gin.data,1,gra.data,1);
	}
};
template<unsigned	R,	unsigned	C,	unsigned	H>
Data<R*C>	self<R,C,H>::gi;
template<unsigned	R,	unsigned	C,	unsigned	H>
Data<2*R*C>	self<R,C,H>::du;
template<unsigned	R,	unsigned	C,	unsigned	H>
struct	wyGPT{
	self<R,C,H>	a;
	sexy<R,C,H>	b;
	self<R,C,H>	c;
	Data<R*C>	&out=c.out;
	void	save(FILE	*F){	a.save(F);	b.save(F);	c.save(F);	}
	void	load(FILE	*F){	a.load(F);	b.load(F);	c.load(F);	}
	unsigned	size(void){	return	a.size()+b.size()+c.size();	}
	uint64_t	flop(void){	return	a.flop()+b.flop()+c.flop();	}
	void	fw(Data<R*C>	&inp){
		a.fw(inp);
		b.fw(a.out);
		c.fw(b.out);
	}
	void	bk(Data<R*C>	&inp,	Data<R*C>	&gin,	Data<R*C> &gra){
		c.bk(b.out,gin,gra);
		b.bk(a.out,gra,gin);
		a.bk(inp,gin,gra);
	}
};
__global__	void	_emb(unsigned	R,	unsigned	C,	uint8_t	*inp,	float	*out){
	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x,	r=id%R,	c=(id/R)%C;	
	out[id]=(_wyhash64(inp[c],r)&1)*2-1.0f;
}
__global__	void	dlossf(unsigned	C,	unsigned	O,	float	*a,	uint8_t	*x,	float	*y){
	float	loss=0;
	for(unsigned	i=0;	i<C;	i++){
		loss-=logf(fmaxf(a[i*O+x[i+1]],FLT_MIN));
		a[i*O+x[i+1]]-=1;
	}
	*y=loss;
}
template<unsigned	C,	unsigned	E,	unsigned	D,	unsigned	H,	unsigned	O>
struct	Neanderthal{
private:
	float	*ret;
	uint8_t	*data;
	Data<E*C>	n0g,trag[2];
public:
	uint64_t	srng=time(NULL);
	Data<E*C>	emb;
	wyGPT<E,C,H>	tra[D];
	layernorm<E,C,1>	n1;
	linear<E,O,C>	ou;
	Neanderthal(){	cudaMallocManaged(&data,	C+1);	cudaMallocManaged(&ret,	sizeof(float));	}
	~Neanderthal(){	cudaFree(data);	cudaFree(ret);	}
	bool	save(const	char	*F){
		FILE	*f=fopen(F,"wb");	if(f==NULL)	return	false;
		unsigned	x;
		x=C;	fwrite(&x,4,1,f);
		x=E;	fwrite(&x,4,1,f);
		x=D;	fwrite(&x,4,1,f);
		x=H;	fwrite(&x,4,1,f);
		x=O;	fwrite(&x,4,1,f);		
		for(unsigned	i=0;	i<D;	i++)	tra[i].save(f);
		ou.save(f);	fclose(f);	return	true;
	}
	bool	load(const	char	*F){
		FILE	*f=fopen(F,"rb");	if(f==NULL)	return	false;
		unsigned	x;
		if(fread(&x,4,1,f)!=1||x!=C)	fprintf(stderr,"C=%u\n",x);
		if(fread(&x,4,1,f)!=1||x!=E)	fprintf(stderr,"E=%u\n",x);
		if(fread(&x,4,1,f)!=1||x!=D)	fprintf(stderr,"D=%u\n",x);
		if(fread(&x,4,1,f)!=1||x!=H)	fprintf(stderr,"H=%u\n",x);
		if(fread(&x,4,1,f)!=1||x!=O)	fprintf(stderr,"O=%u\n",x);	
		for(unsigned	i=0;	i<D;	i++)	tra[i].load(f);
		ou.load(f);	fclose(f);	return	true;
	}
	unsigned	size(void){	return	tra[0].size()*D+ou.size();	}
	float	train(uint8_t	*text,	uint64_t	len){
		cudaMemcpy(data,text+(wyrand(&srng)%(len-C)),C+1,cudaMemcpyHostToDevice);
		_emb<<<E*C/16,16>>>(E,C,data,emb.data);
		for(unsigned	d=0;	d<D;	d++)	tra[d].fw(d?tra[d-1].out:emb);
		n1.fw(tra[D-1].out);	ou.fw(tra[D-1].out);
		_softmaxf<<<C/16,16>>>(O,ou.out.data);
		dlossf<<<1,1>>>(C,O,ou.out.data,data,ret);
		ou.bk(tra[D-1].out,ou.out,n0g);	n1.bk(tra[D-1].out,n0g);
		for(unsigned	d=D-1;	d<D;	d--)	tra[d].bk(d?tra[d-1].out:emb,d<D-1?trag[(d+1)%2]:n0g,trag[d%2]);
		cudaDeviceSynchronize();	return	*ret;
	}
};
#include	"config"
using	namespace	std;
Neanderthal<context,embed,depth,heads,voca>	model;
void	document(void){
	cerr<<"usage:	train [options] input1.txt [input2.txt input3.txt...]\n";
	cerr<<"\t-i:	input model=NULL\n";
	cerr<<"\t-o:	output model=model\n";
	cerr<<"\t-s:	trained sample=0\n";
	cerr<<"\t-b:	benchmark only=off\n";
	exit(0);
}
struct	Dataset{
	string	name;
	uint8_t	*ptr;
	int	fd;
	struct	stat	sb;
	double	weight;
};
int	main(int	ac,	char	**av){
	cublasCreate(&handle);	string	in,out="model";	int	opt,bench=0;	uint64_t	training=0;
	while((opt=getopt(ac,	av,	"i:o:s:b"))>=0){
		switch(opt){
		case	'i':	in=optarg;	break;
		case	'o':	out=optarg;	break;
		case	's':{	training=atoi(optarg);	training<<=20;	}	break;
		case	'b':	bench=1;	model.srng=0;	break;
		default:	document();
		}
	}
	if(ac<optind+1){	document();	return	0;	}
	vector<Dataset>	ds;	ds.resize(ac-optind);	double	sum_weight=0;
	for(int	i=optind;	i<ac;	i++){
		int	j=i-optind;
		ds[j].name=av[i];	ds[j].fd=open(av[i],	O_RDONLY);	fstat(ds[j].fd,	&ds[j].sb);
		ds[j].ptr=(uint8_t*)mmap(NULL,	ds[j].sb.st_size,	PROT_READ,	MAP_SHARED,	ds[j].fd,	0);
		sum_weight+=(ds[j].weight=1);
		cerr<<av[i]<<'\t'<<ds[j].sb.st_size/1024.0f/1024<<'\t'<<ds[j].weight<<'\n';
	}
	cerr.precision(4);	cerr.setf(ios::fixed);
	double	loss0=FLT_MAX/2,	loss;	timeval	beg,	end;	
	size_t	para=model.size();	cerr<<"para\t"<<para<<'\n';	
	if(in.size())	model.load(in.c_str());
	for(;;){
		loss=0;	gettimeofday(&beg,NULL);	vector<double>	vl(ds.size()),vn(ds.size());	
		for(size_t	i=0;	i<fullbatch;	i++){
			eta=2.0f/sqrtf(log1pf(para)*para+training);	training+=context;
			double	ran=wy2u01(wyrand(&prng))*sum_weight,sum=0;
			size_t	r=ds.size()-1;
			for(size_t	j=0;	j<ds.size();	j++){	
				sum+=ds[j].weight;
				if(sum>=ran){	r=j;	break;	}
			}
			double	l=model.train(ds[r].ptr,ds[r].sb.st_size);
			loss+=l;	vl[r]+=l;	vn[r]+=context;
		}
		loss/=context*fullbatch;
		if(!bench){	if(loss<loss0+0.02)	model.save(out.c_str());	else	break;	}
		loss0=loss;	gettimeofday(&end,NULL);
		cerr<<(training>>20);
		for(size_t	i=0;	i<ds.size();	i++)	cerr<<'\t'<<vl[i]/vn[i];
		cerr<<'\t'<<(end.tv_sec-beg.tv_sec+1e-6*(end.tv_usec-beg.tv_usec))<<'\n';
	}
	for(int	j=0;	j<ds.size();	j++){	munmap(ds[j].ptr,ds[j].sb.st_size);	close(ds[j].fd);	}
	cublasDestroy(handle);
	return	0;
}
