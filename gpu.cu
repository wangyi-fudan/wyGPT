#include	<cuda_runtime.h>
#include	<unordered_map>
#include	<cublas_v2.h>
#include	<sys/time.h>
#include	<algorithm>
#include	<iostream>
#include	<unistd.h>
#include	<sstream>
#include	<cstdint>
#include	<cfloat>
#include	<cstdio>
#include	<vector>
using	namespace	std;
cublasHandle_t	handle;
uint64_t	prng=time(NULL);
static inline uint64_t wyrand(uint64_t	*seed){	*seed+=0xa0761d6478bd642full;	uint64_t  see1=*seed^0xe7037ed1a0b428dbull;	see1*=(see1>>32)|(see1<<32);	return	(*seed*((*seed>>32)|(*seed<<32)))^((see1>>32)|(see1<<32));	}
static inline double wy2u01(uint64_t r){	const double _wynorm=1.0/(1ull<<52);	return (r>>12)*_wynorm;	}
void _wymum(uint64_t *A,	uint64_t *B){	uint64_t	hh=(*A>>32)*(*B>>32), hl=(*A>>32)*(uint32_t)*B, lh=(uint32_t)*A*(*B>>32), ll=(uint64_t)(uint32_t)*A*(uint32_t)*B;	*A=((hl>>32)|(hl<<32))^hh;	*B=((lh>>32)|(lh<<32))^ll;	}
uint64_t	_wyhash64(uint64_t	A,	uint64_t	B){	A^=0xa0761d6478bd642full;	B^=0xa0761d6478bd642full;	_wymum(&A,&B);	A^=0xa0761d6478bd642full;	B^=0xa0761d6478bd642full;	_wymum(&A,&B);	return	A^B;	}
template<unsigned	N>
struct	Data16{
	__nv_bfloat16	*data;
	Data16(){	cudaMallocManaged(&data,	N*sizeof(__nv_bfloat16));	}
	~Data16(){	cudaFree(data);	}
	void	zero(void){	cudaMemset(data,	0,	N*sizeof(__nv_bfloat16));	}
	void	load(FILE	*F){	if(fread(data,N*2,1,F)!=1)	return;	}	
};
__global__	void	_s16(unsigned	N,	float	*w, __nv_bfloat16	*g){	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x;	if(id<N)	g[id]=__float2bfloat16(w[id]);	}
__global__	void	_l16(unsigned	N,	float	*w, __nv_bfloat16	*g){	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x;	if(id<N)	w[id]=__bfloat162float(g[id]);	}
template<unsigned	N>
struct	Data{
	static	Data16<N>	tmp;
	float	*data;
	Data(){	cudaMallocManaged(&data,	N*sizeof(float));	}
	~Data(){	cudaFree(data);	}
	void	zero(void){	cudaMemset(data,	0,	N*sizeof(float));	}
	void	load(FILE	*F){	if(fread(tmp.data,N*2,1,F)!=1){	return;	}	_l16<<<(N+15)/16,16>>>(N,data,tmp.data);	cudaDeviceSynchronize();	}
};
template<unsigned	N>
Data16<N>	Data<N>::tmp;
template<unsigned	R0,	unsigned	R1>
struct	linear{
	Data16<R0*R1>	wei;
	Data<R1>	out;
	void	load(FILE	*F){	wei.load(F);	}
	void	fw(Data<R0>	&inp){
		float	alf=1/sqrtf(R0),	bet=0;
		_s16<<<R0/16,16>>>(R0,inp.data,inp.tmp.data);
		cublasTSSgemvStridedBatched(handle,CUBLAS_OP_T,R0,R1,&alf,wei.data,R0,0,inp.tmp.data,1,0,&bet,out.data,1,0,1);
	}
};
__global__	void	_layernorm(unsigned	R,	float	*inp,	unsigned	H){
	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x,	r=R/H;
	float	sum=0,	nor=0,	*in=inp+id*r;	
	for(unsigned	i=0;	i<r;	i+=4){	float4*	t=(float4*)(in+i);	sum+=t->x+t->y+t->z+t->w;	nor+=t->x*t->x+t->y*t->y+t->z*t->z+t->w*t->w;	}
	sum/=r;	nor=sqrtf(r/fmaxf(nor-sum*sum*r,1e-18f));
	for(unsigned	i=0;	i<r;	i+=4){	float4	*t=(float4*)(in+i);	*t=make_float4((t->x-sum)*nor,(t->y-sum)*nor,(t->z-sum)*nor,(t->w-sum)*nor);	}
}
void	softmax(unsigned	R,	float	*inp){
	float	sum=0,	ma=-FLT_MAX;
	for(unsigned	i=0;    i<R;    i++)	ma=fmaxf(inp[i],ma);
	for(unsigned	i=0;	i<R;	i++)	sum+=(inp[i]=expf(inp[i]-ma));
	for(unsigned	i=0;	i<R;	i++)	inp[i]/=sum;
}
__global__	void	_sexyfp(unsigned	C,	unsigned	para,	unsigned	col,	float	*att,	float	*pe){
	unsigned	id=blockIdx.x*blockDim.x+threadIdx.x,	j=id%C,	h=id/C,	i=(j+1+col)%C;
	if(j<para)	att[h*C+i]=0;
	else	att[h*C+i]=expf(pe[h*C+C-1-j]+att[h*C+i]);
}
__global__	void	_sexyfsuv(float	*u,	float	*v){
	unsigned	id=(blockIdx.x*blockDim.x+threadIdx.x)<<2;
	float4	*u4=(float4*)(u+id),	*v4=(float4*)(v+id);
	*v4=make_float4(u4->x*v4->x,u4->y*v4->y,u4->z*v4->z,u4->w*v4->w);
}
__global__	void	_sexyadd(float	*u,	float	*v){
	unsigned	id=(blockIdx.x*blockDim.x+threadIdx.x)<<2;
	float4	*u4=(float4*)(u+id),	*v4=(float4*)(v+id);
	*u4=make_float4(u4->x+v4->x,u4->y+v4->y,u4->z+v4->z,u4->w+v4->w);
}
template<unsigned	R,	unsigned	C,	unsigned	H>
struct	sexy{
	static	Data<R>	va;
	static	Data<H*C>	a;
	Data16<R*C>	k0,k1;
	Data<H*C>	pe;
	linear<R,4*R>	x;
	linear<R,R>	o;
	Data<R>	&out=o.out;
	sexy(){	k0.zero();	k1.zero();	}
	void	load(FILE* F){	 pe.load(F); x.load(F);	o.load(F);	}
	void	fw(Data<R>	&inp,	unsigned	col,	unsigned	para){
		x.fw(inp);	_layernorm<<<4*H,1>>>(4*R,x.out.data,4*H);
		_s16<<<R/16,16>>>(R,x.out.data,k0.data+col*R);
		_s16<<<R/16,16>>>(R,x.out.data+R,x.out.tmp.data+R);
		_s16<<<R/16,16>>>(R,x.out.data+2*R,k1.data+col*R);
		float	alf=1/sqrtf(R/H),	alf1=1,bet=0;
		cublasTSSgemvStridedBatched(handle,CUBLAS_OP_T,R/H,C,&alf,k0.data,R,R/H,x.out.tmp.data+R,1,R/H,&bet,a.data,1,C,H);
		_sexyfp<<<C*H/16,16>>>(C,para,col,a.data,pe.data);
		_s16<<<H*C/16,16>>>(H*C,a.data,a.tmp.data);
		cublasTSSgemvStridedBatched(handle,CUBLAS_OP_N,R/H,C,&alf1,k1.data,R,R/H,a.tmp.data,1,C,&bet,va.data,1,R/H,H);
		_sexyfsuv<<<R/4/4,4>>>(x.out.data+3*R,va.data);
		_layernorm<<<H,1>>>(R,va.data,H);	o.fw(va);
		_sexyadd<<<R/4/4,4>>>(o.out.data,inp.data);
	}
};
template<unsigned	R,	unsigned	C,	unsigned	H>
Data<R>	sexy<R,C,H>::va;
template<unsigned	R,	unsigned	C,	unsigned	H>
Data<H*C>	sexy<R,C,H>::a;
__global__	void	_selffsuv(unsigned	S,	float	*u,	float	*o){
	unsigned	id=(blockIdx.x*blockDim.x+threadIdx.x)<<2;
	float4	*u4=(float4*)(u+id),	*v4=(float4*)(u+S+id),	*o4=(float4*)(o+id);
	*o4=make_float4(u4->x*v4->x,u4->y*v4->y,u4->z*v4->z,u4->w*v4->w);
}
template<unsigned	R,	unsigned	C,	unsigned	H>
struct	self{
	static	Data<R>	tmp;
	linear<R,2*R>	u;
	linear<R,R>	o;
	Data<R>	&out=o.out;
	void	load(FILE* F){	 u.load(F);	o.load(F);	}
	void	fw(Data<R>	&inp){
		u.fw(inp);	_layernorm<<<2*H,1>>>(2*R,u.out.data,2*H);
		_selffsuv<<<R/4/4,4>>>(R,u.out.data,tmp.data);
		_layernorm<<<H,1>>>(R,tmp.data,H);	o.fw(tmp);
		_sexyadd<<<R/4/4,4>>>(o.out.data,inp.data);
	}
};
template<unsigned	R,	unsigned	C,	unsigned	H>
Data<R>	self<R,C,H>::tmp;
template<unsigned	R,	unsigned	C,	unsigned	H>
struct	wyGPT{
	self<R,C,H>	a;
	sexy<R,C,H>	b;
	self<R,C,H>	c;
	Data<R>	&out=c.out;
	void	load(FILE* F){	 a.load(F);	b.load(F);	c.load(F);	}
	void	fw(Data<R>	&inp,	unsigned	col,	unsigned	para){
		a.fw(inp);
		b.fw(a.out,col,para);
		c.fw(b.out);
	}
};
template<unsigned	C,	unsigned	E,	unsigned	D,	unsigned	H,	unsigned	O>
struct	Neanderthal{
	unsigned	curr=0;
	Data<E>	emb;
	wyGPT<E,C,H>	tra[D];
	linear<E,O>	out;
	float	vs[O];
	bool	load(const	char	*F){
		FILE* f=fopen(F, "rb");
		if(f==NULL)	return	false;
		unsigned	x;		
		if(fread(&x,4,1,f)!=1||x!=C)	fprintf(stderr,"C=%u\n",x);
		if(fread(&x,4,1,f)!=1||x!=E)	fprintf(stderr,"E=%u\n",x);
		if(fread(&x,4,1,f)!=1||x!=D)	fprintf(stderr,"D=%u\n",x);
		if(fread(&x,4,1,f)!=1||x!=H)	fprintf(stderr,"H=%u\n",x);
		if(fread(&x,4,1,f)!=1||x!=O)	fprintf(stderr,"O=%u\n",x);	
		for(unsigned i=0; i<D; i++)	tra[i].load(f);
		out.load(f);	fclose(f);
		return	true;
	}
	uint8_t	sample(uint8_t	*x,	uint8_t	*p){
		unsigned	para=p+C-1>=x?p+C-1-x:0;
		for(unsigned	r=0;	r<E;	r++)	emb.data[r]=(_wyhash64(*x,r)&1)*2-1.0f;
		for(unsigned	d=0;	d<D;	d++)	tra[d].fw(d?tra[d-1].out:emb,curr,para);
		_layernorm<<<1,1>>>(E,tra[D-1].out.data,1);	out.fw(tra[D-1].out);
		cudaDeviceSynchronize();
		for(unsigned    i=0;    i<O;    i++)	out.out.data[i]=M_SQRT2*(out.out.data[i]-vs[i]);
		softmax(O,out.out.data);
		double	sum=0;	for(unsigned    i=0;    i<O;    i++)	sum+=(out.out.data[i]=fmaxf(out.out.data[i]-1.0f/O,0));
		double  ran=wy2u01(wyrand(&prng))*sum,  sum1=0; uint16_t        ret=0;
		for(size_t      i=0;    i<O;    i++){   sum1+=out.out.data[i];    if(sum1>=ran){  ret=i;  break;  }       }
		curr=(curr+1)%C;	return	ret;
	}
	string	generate(string	inp,	unsigned	n){
		if(!inp.size())	return	"";
		vector<uint8_t>	s;	uint8_t	c;
		for(unsigned	i=0;	i<inp.size()&&i<n;	i++){	
			s.push_back(inp[i]);	
			memset(vs,0,sizeof(float)*O);
			for(size_t	k=0;	k<s.size();	k++){
				unsigned	l=1;
				while(l<=k&&s[k-l]==s[s.size()-l])	l++;
				vs[s[k]]+=(expf(l/M_E)-1)/(s.size()-k);
			}
			c=sample(s.data()+s.size()-1,s.data());	
		}
		while(s.size()<n){	
			s.push_back(c);
			memset(vs,0,sizeof(float)*O);
			for(size_t	k=0;	k<s.size();	k++){
				unsigned	l=1;
				while(l<=k&&s[k-l]==s[s.size()-l])	l++;
				vs[s[k]]+=(expf(l/M_E)-1)/(s.size()-k);
			}
			c=sample(s.data()+s.size()-1,s.data());	
		}
		string	ret(s.begin(),s.end());
		return	ret;
	}
	float	probability(const	uint8_t	*x,	const	uint8_t	*p){
		unsigned	para=p+C-1>=x?p+C-1-x:0;
		for(unsigned	r=0;	r<E;	r++)	emb.data[r]=(_wyhash64(*x,r)&1)*2-1.0f;
		for(unsigned	d=0;	d<D;	d++)	tra[d].fw(d?tra[d-1].out:emb,curr,para);
		_layernorm<<<1,1>>>(E,tra[D-1].out.data,1);	out.fw(tra[D-1].out);
		cudaDeviceSynchronize();	
		softmax(O,out.out.data);	curr=(curr+1)%C;	return	out.out.data[x[1]];
	}
	float	evaluate(string	inp){
		double	loss=0;
		for(unsigned	i=0;	i+1<inp.size();	i++)	
			loss-=logf(fmaxf(probability((uint8_t*)inp.data()+i,(uint8_t*)inp.data()),FLT_MIN));
		return	inp.size()<2?0:loss/(inp.size()-1);
	}
};
#include	"config"
int	main(int	ac,	char	**av){
	cublasCreate(&handle);
	Neanderthal<context,embed,depth,heads,voca>	model;
	string	model_file="model";
	int	opt;
	while((opt=getopt(ac,	av,	"m:"))>=0){
		switch(opt){
		case	'm':	model_file=optarg;	break;
		}
	}
	if(!model.load(model_file.c_str())){	fprintf(stderr,"fail to load %s\n",model_file.c_str());	return	0;	}
	timeval	beg,end;	gettimeofday(&beg,NULL);
	cout<<model.generate(av[optind],context)<<'\n';	// the second parameter can be arbitary long
	gettimeofday(&end,NULL);
	cerr<<end.tv_sec-beg.tv_sec+1e-6*(end.tv_usec-beg.tv_usec)<<'\n';
	cublasDestroy(handle);
	return	0;
}
