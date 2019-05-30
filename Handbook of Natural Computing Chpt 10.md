# 개요
* 3세대 뉴럴넷  
* 뇌에서 실제로 일어나는 연산 과정에 더 가까움  
* 뉴런 사이 시냅스에서 발생하는 spike의 타이밍을 모델링  
* 기존 뉴럴넷 보다 computational power도 더 좋음 (capacity가 지수함수적으로 증가하는 듯?)  
* 현재 관건은 snn의 장점을 유지하면서도 효율적인 학습 방법을 찾는 것  


챕터 개요
1. Spiking 뉴런의 역사
2. 지금까지 쓰이고 있는 (spiking)뉴런과 시냅스 모델들
3. snn의 computational power(표현 범위)
4. snn학습 방법(아마도 왜 학습이 어려운지)
5. 실제로 적용하고 구현해보기, 시뮬레이션 프레임워크 등



# 1. From Natural Computing to Artificial Neural Networks
## 1.1 Traditional Neural Networks
AI 분야는 사람 뇌(뉴럴넷)에서 일어나는 가장 핵심적인 연산 과정이 뭔지 알아내려고 시도해 왔음  

#### McCulloch and Pitts(1943)가 맨 처음 제안한 (사람의)뉴럴넷 기반 모델  
* 그냥 바이너리 뉴런. actv. func.이 sigmoid도 아니고 그냥 step func.  
* 입력으로 들어오는 값의 weighted sum이 기준치보다 높냐 낮냐에 따라 출력을 결정함  
  
그 이후로 이것 저것 perceptron이랑 sigmoid a.f. 쓴 mlp도 나옴  
weighted sum은 <X, W> (X와 W의 내적)이라 생각할 수 있는데,  
이걸 X와 W가 얼마나 닮았냐(가깝냐)로 해석할 수 있음  
같은 맥락에서 둘의 가까운 정도를 그냥 거리 |X - W|로 정의해보려는 시도도 있었음(Kohonen(1982), Van Hulle(2000))  
단순한 연산 단위들이 연결된 것만으로 입력과 출력 사이 관계를 표현하는 수많은 수학적 함수들을 흉내냄
뉴런 사이의 가중치를 정해주는 알고리즘(학습 규칙)을 통해 뉴럴넷이 관계를 "학습"하게 할 수 있음

#### 학습 규칙 분류
Supervised learning: 경사하강법 포함(ex. 백프롭)  
Unsupervised learning: 많은 아이디어가 Hebb(1949)의 시냅스 가소성(synaptic plasticity)에 기반을 둠  

Hebb이 시냅스 가소성과 관련해 한 말
> 뉴런A가 뉴런B를 흥분시키기에 충분히 가깝거나 지속적/영구적인 기여를 한다면, A가 B에게 더 효율적으로 영향을 끼칠 수 있도록 두 세포 중 어느 한쪽이나 양쪽 전부에게 변화가 일어난다.

저 말에 영향을 받은 unsupervised 학습 규칙들을 Hebbian rules이라 부름(ex. Hopfield's(1982))  
ann이 많은 분야에서 강력한 engineering도구로 쓰이고 있고 이론적인 분야에서도 강력함  
예시로 calculability, complecity, capacity, regularization theory가 있음 (~~레퍼런스만 달려있어서 어느 맥락에서 나온 얘긴지는 잘 모르겠음~~)
그럼에도 불구하고 기존의 신경망은 본질적인 한계를 가짐: 많은 양의 데이터를 다루는 것이나 환경 변화에 빠르게 적응하는 것
생물학적인 신경에서 일어나는 과정에 비하면 ann은 매우 제한적임
