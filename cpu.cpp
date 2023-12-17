#include	<x86intrin.h>
#include	<sys/time.h>
#include	<algorithm>
#include	<iostream>
#include	<unistd.h>
#include	<cstring>
#include	<sstream>
#include	<cfloat>
#include	<cstdio>
#include	<vector>
#include	<cmath>
#include	<ctime>
#include	<omp.h>
using	namespace	std;
uint64_t	prng=time(NULL);
static inline uint64_t wyrand(uint64_t *seed){	*seed+=0xa0761d6478bd642full;	uint64_t  see1=*seed^0xe7037ed1a0b428dbull;	see1*=(see1>>32)|(see1<<32);	return  (*seed*((*seed>>32)|(*seed<<32)))^((see1>>32)|(see1<<32));}
static inline double wy2u01(uint64_t r){ const double _wynorm=1.0/(1ull<<52); return (r>>12)*_wynorm;}
static inline float wy2gau(uint64_t r){ const float	_wynorm=1.0/(1ull<<20); return ((r&0x1fffff)+((r>>21)&0x1fffff)+((r>>42)&0x1fffff))*_wynorm-3.0f;}
static inline void _wymum(uint64_t *A,	uint64_t *B){	uint64_t	hh=(*A>>32)*(*B>>32), hl=(*A>>32)*(uint32_t)*B, lh=(uint32_t)*A*(*B>>32), ll=(uint64_t)(uint32_t)*A*(uint32_t)*B;	*A=((hl>>32)|(hl<<32))^hh;	*B=((lh>>32)|(lh<<32))^ll;}
static inline uint64_t	_wyhash64(uint64_t	A,	uint64_t	B){	A^=0xa0761d6478bd642full; B^=0xa0761d6478bd642full;	_wymum(&A,&B);	A^=0xa0761d6478bd642full; B^=0xa0761d6478bd642full;	_wymum(&A,&B);	return	A^B;}
template<unsigned	N>
struct	Data{
	float	*data;
	Data(){	data=(float*)aligned_alloc(64,N*sizeof(float));	}
//	Data(){	data=(float*)_aligned_malloc(N*sizeof(float),64);	}
	~Data(){	free(data);	}
//	~Data(){	_aligned_free(data);	}
	bool	load(FILE	*F){	
		uint16_t	*f16=new	uint16_t[N];	
		if(fread(f16,N*sizeof(uint16_t),1,F)!=1)	return	false;
		for(unsigned	i=0;	i<N;	i++){
			uint16_t	*p=(uint16_t*)(data+i);
			p[0]=0;	p[1]=f16[i];
		}
		delete	[]	f16;
		return	true;
	}
};
template<unsigned   R, unsigned   C>
class vnni{
public:
   float scale[C];
   short  *data;
   vnni(){ data=(short*)aligned_alloc(64,R*C*2);  memset(data,0,R*C*2);	}
//	vnni(){	data=(short*)_aligned_malloc(R*C*2,64);	memset(data,0,R*C*2);	}
	~vnni(){   free(data); }
//	~vnni(){   _aligned_free(data); }   
   void  float2vnni(float  *p,   unsigned   c){
      short  *q=data+c*R;  float ma=0;
      for(unsigned  i=0;  i<R;  i++)  ma=fmaxf(fabsf(p[i]),ma);
      scale[c]=ma=sqrtf(2147483647/R)/ma;
      for(unsigned  i=0;  i<R;  i++)  q[i]=roundf(p[i]*ma);
   }
};
typedef	short	v8hi	__attribute__	((__vector_size__	(16)));
typedef	int	v4si	__attribute__	((__vector_size__	(16)));

static	inline	int	_mm_reduce_add_epi32(v4si	x){	int *p=(int*)&x;	return	p[0]+p[1]+p[2]+p[3];	}
template<unsigned   R, unsigned   C0,   unsigned   C1>
float dot(vnni<R,C0>   &ma,  unsigned   ca,   vnni<R,C1> &mb,  unsigned   cb){
	short  *p=ma.data+ca*R,*q=mb.data+cb*R;	v4si	v={};
	for(unsigned  i=0;  i<R;  i+=8)	v=__builtin_ia32_paddd128(v,__builtin_ia32_pmaddwd128(*(v8hi*)(p+i),*(v8hi*)(q+i)));    
	return   _mm_reduce_add_epi32(v)/(ma.scale[ca]*mb.scale[cb]);
};
template<unsigned   R, unsigned   C0,   unsigned   C1>
float dot(vnni<R,C0>   &ma,  unsigned   ca,   vnni<R,C1> &mb,  unsigned   cb,	unsigned	h,	unsigned	H){
	short  *p=ma.data+ca*R+h*(R/H),*q=mb.data+cb*R+h*(R/H);	v4si	v={};
	for(unsigned  i=0;  i<R/H;  i+=8)	v=__builtin_ia32_paddd128(v,__builtin_ia32_pmaddwd128(*(v8hi*)(p+i),*(v8hi*)(q+i)));    
	return   _mm_reduce_add_epi32(v)/(ma.scale[ca]*mb.scale[cb]);
};

template<unsigned   R, unsigned   C>
class vnnip{
public:
   const	float scale=32767;
   short  *data;
   vnnip(){ data=(short*)aligned_alloc(64,R*C*2);  memset(data,0,R*C*2);	}
//	vnnip(){	data=(short*)_aligned_malloc(R*C*2,64);	memset(data,0,R*C*2);	} 
	~vnnip(){   free(data); }
//	~vnnip(){   _aligned_free(data); }
   void  float2vnni(float  *p,   unsigned   c){
      short  *q=data+c*R;
      for(unsigned  i=0;  i<R;  i++)  q[i]=roundf(p[i]*scale);
   }
};
template<unsigned   R, unsigned   C>
class vnnit{
public:
   const float scale=32767/sqrtf(C);
   short  *data;
   vnnit(){ data=(short*)aligned_alloc(64,R*C*2);  memset(data,0,R*C*2);	}
//	vnnit(){	data=(short*)_aligned_malloc(R*C*2,64);	memset(data,0,R*C*2);	}
	~vnnit(){   free(data); }
//	~vnnit(){   _aligned_free(data); }
   void  float2vnni(float  *p,   unsigned   r){
      for(unsigned  i=0;  i<C;  i++)	 data[i*R+r]=roundf(p[i]*scale);
   }
};
template<unsigned   R, unsigned   C0,   unsigned   C1>
float dot(vnnit<R,C0>   &ma,  unsigned   ca,   vnnip<R,C1> &mb,  unsigned   cb){
	short  *p=ma.data+ca*R, *q=mb.data+cb*R;	v4si	v={};
	for(unsigned  i=0;  i<R;  i+=8)	v=__builtin_ia32_paddd128(v,__builtin_ia32_pmaddwd128(*(v8hi*)(p+i),*(v8hi*)(q+i)));    
	return   _mm_reduce_add_epi32(v)/(ma.scale*mb.scale);
}
template<unsigned	R0,	unsigned	R1>
struct	linear{
	vnni<R0,R1>	vwei;
	vnni<R0,1>	vinp;
	Data<R1>	out;
	void	load(FILE	*F){
		Data<R0*R1>	wei;	wei.load(F);
		for(unsigned	i=0;	i<R1;	i++)	vwei.float2vnni(wei.data+i*R0,i);
	}
	void	fw(Data<R0*1>	&inp,	unsigned	col=0){
		float	*ou=out.data+col*R1,	alf=1/sqrtf(R0);
		vinp.float2vnni(inp.data,0);
		#pragma omp parallel for
		for(unsigned	i=0;	i<R1;	i++)	ou[i]=alf*dot(vinp,0,vwei,i);
	}
};
template<unsigned	R,	unsigned	H>
void	layernorm(Data<R>	&inp){
	unsigned	r=R/H;
	#pragma omp parallel for
	for(unsigned	h=0;	h<H;	h++){
		float	sum=0,	nor=0,	*p=inp.data+h*r;	
		for(unsigned	i=0;	i<r;	i++){	sum+=p[i];	nor+=p[i]*p[i];	}
		sum/=r;	nor=fmaxf(nor-sum*sum*r,1e-18f);	nor=sqrtf(r/nor);
		for(unsigned	i=0;	i<r;	i++)	p[i]=(p[i]-sum)*nor;
	}	
}
template<unsigned	R>
void	softmax(Data<R>	&inp){
	float	ma=-FLT_MAX,	sum=0;
	for(unsigned	i=0;	i<R;	i++)	if(inp.data[i]>ma)	ma=inp.data[i];
	for(unsigned	i=0;	i<R;	i++)	sum+=(inp.data[i]=expf(inp.data[i]-ma));
	for(unsigned	i=0;	i<R;	i++)	inp.data[i]/=sum;
}
template<unsigned	R,	unsigned	C,	unsigned	H>
struct	sexy{
	vnni<R/H,1>	vq;
	vnni<R/H,C>	vk[H];
	vnnip<C,1>	va;
	vnnit<C,R>	pn;
	Data<H*C>	pe;
	linear<R,4*R>	x;
	linear<R,R>	o;
	Data<R>	tmp,&out=o.out;
	void	load(FILE	*F){	pe.load(F);	x.load(F);	o.load(F);	}
	void	fw(Data<R>	&inp,	unsigned	col,	unsigned	para){
		x.fw(inp);	layernorm<4*R,4*H>(x.out);	pn.float2vnni(x.out.data+2*R,col);
		for(unsigned	h=0;	h<H;	h++){
			vk[h].float2vnni(x.out.data+h*(R/H),col);
			vq.float2vnni(x.out.data+R+h*(R/H),0);
			float	alf=1/sqrtf(R/H),	a[C]={},	sum=0;
			#pragma omp parallel for
			for(unsigned	j=para;	j<C;	j++){	unsigned	i=(j+1+col)%C;	a[i]=expf(dot(vk[h],i,vq,0)*alf+pe.data[h*C+C-1-j]);	}
			for(unsigned	i=0;	i<C;	i++)	sum+=a[i];
			for(unsigned	i=0;	i<C;	i++)	a[i]/=sum;
			va.float2vnni(a,0);
			#pragma omp parallel for
			for(unsigned	i=0;	i<R/H;	i++)	tmp.data[h*(R/H)+i]=dot(pn,h*(R/H)+i,va,0);
		}	
		for(unsigned	i=0;	i<R;	i++)	tmp.data[i]*=x.out.data[3*R+i];
		layernorm<R,H>(tmp);	o.fw(tmp);
		for(unsigned	i=0;	i<R;	i++)	o.out.data[i]+=inp.data[i];
	}
};
template<unsigned	R,	unsigned	C,	unsigned	H>
struct	self{
	static	Data<R>	tmp;
	linear<R,2*R>	u;
	linear<R,R>	o;
	Data<R>	&out=o.out;
	void	load(FILE* F){	 u.load(F);	o.load(F);	}
	void	fw(Data<R>	&inp){
		u.fw(inp);	layernorm<2*R,2*H>(u.out);
		for(unsigned	i=0;	i<R;	i++)	tmp.data[i]=u.out.data[i]*u.out.data[R+i];
		layernorm<R,H>(tmp);	o.fw(tmp);
		for(unsigned	i=0;	i<R;	i++)	o.out.data[i]+=inp.data[i];	
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
		layernorm<E,1>(tra[D-1].out);	out.fw(tra[D-1].out);
		for(unsigned    i=0;    i<O;    i++)	out.out.data[i]=M_SQRT2*(out.out.data[i]-vs[i]);
		softmax<O>(out.out);
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
		layernorm<E,1>(tra[D-1].out);	out.fw(tra[D-1].out);
		softmax<O>(out.out);	curr=(curr+1)%C;	return	out.out.data[x[1]];
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
	size_t	threads=omp_get_num_procs();
	Neanderthal<context,embed,depth,heads,voca>	model;
	string	model_file="model";
	int	opt;
	while((opt=getopt(ac,	av,	"m:t:"))>=0){
		switch(opt){
		case	'm':	model_file=optarg;	break;
		case	't':	threads=atoi(optarg);	break;
		}
	}
	omp_set_num_threads(threads);
	if(!model.load(model_file.c_str())){	fprintf(stderr,"fail to load %s\n",model_file.c_str());	return	0;	}
	timeval	beg,end;	gettimeofday(&beg,NULL);
	cout<<model.generate(av[optind],context)<<'\n';	//	the second parameter can be arbitary long
	gettimeofday(&end,NULL);
	cerr<<end.tv_sec-beg.tv_sec+1e-6*(end.tv_usec-beg.tv_usec)<<'\n';
	return	0;
}
