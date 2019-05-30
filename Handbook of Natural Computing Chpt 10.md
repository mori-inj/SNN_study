# 0. 개요
### Spiking Neural Network
* 3세대 뉴럴넷  
* 뇌에서 실제로 일어나는 연산 과정에 더 가까움  
* 뉴런 사이 시냅스에서 발생하는 spike의 타이밍을 모델링  
* 기존 뉴럴넷 보다 computational power도 더 좋음 (capacity가 지수함수적으로 증가하는 듯?)  
* 현재 관건은 snn의 장점을 유지하면서도 효율적인 학습 방법을 찾는 것  


### 챕터 개요
1. Spiking 뉴런의 역사
2. 지금까지 쓰이고 있는 (spiking)뉴런과 시냅스 모델들
3. snn의 computational power(표현 범위)
4. snn학습 방법(아마도 왜 학습이 어려운지)
5. 실제로 적용하고 구현해보기, 시뮬레이션 프레임워크 등

&nbsp;  
&nbsp;

# 1. From Natural Computing to Artificial Neural Networks
## 1.1 Traditional Neural Networks
AI 분야는 사람 뇌(뉴럴넷)에서 일어나는 가장 핵심적인 연산 과정이 뭔지 알아내려고 시도해 왔음  

#### McCulloch and Pitts(1943)가 맨 처음 제안한 (사람의)뉴럴넷 기반 모델  
* 그냥 바이너리 뉴런. actv. func.이 sigmoid도 아니고 그냥 step func.  
* 입력으로 들어오는 값의 weighted sum이 기준치보다 높냐 낮냐에 따라 출력을 결정함  
  
그 이후로 이것 저것 perceptron이랑 sigmoid a.f. 쓴 mlp도 나옴

weighted sum은 <X, W> (X와 W의 내적)이라 생각할 수 있는데, 이걸 X와 W가 얼마나 닮았냐(가깝냐)로 해석할 수 있음. 같은 맥락에서 둘의 가까운 정도를 그냥 거리 |X - W|로 정의해보려는 시도도 있었음(Kohonen(1982), Van Hulle(2000))

단순한 연산 단위들이 연결된 것만으로 입력과 출력 사이 관계를 표현하는 수 많은 수학적 함수들을 흉내냄. 뉴런 사이의 가중치를 정해주는 알고리즘(학습 규칙)을 통해 뉴럴넷이 관계를 "학습"하게 할 수 있음

&nbsp;

#### 학습 규칙 분류
* Supervised learning: 경사하강법 포함(ex. 백프롭)  
* Unsupervised learning: 많은 아이디어가 Hebb(1949)의 시냅스 가소성(synaptic plasticity)에 기반을 둠  

##### Hebb이 시냅스 가소성과 관련해 한 말
> 뉴런A가 뉴런B를 흥분시키기에 충분히 가깝거나 지속적/영구적인 기여를 한다면, A가 B에게 더 효율적으로 영향을 끼칠 수 있도록 두 세포 중 어느 한쪽이나 양쪽 전부에게 변화가 일어난다.

저 말에 영향을 받은 unsupervised 학습 규칙들을 Hebbian rules이라 부름(ex. Hopfield's(1982))

ann이 많은 분야에서 강력한 engineering도구로 쓰이고 있고 이론적인 분야에서도 강력함  
이론적인 분야의 예시: calculability, complecity, capacity, regularization theory  
(*레퍼런스만 달려있어서 어느 맥락에서 나온 얘긴지는 잘 모르겠음*)

신경망의 본질적 한계: 많은 양의 데이터 X, 환경 변화에 빠르게 적응 X  
생물학적인 신경에서 일어나는 과정에 비하면 매우 제한적임  

&nbsp;

## 1.2 The Biological Inspiration, Revisited
1. *기존 방식은 시간을 감안 안 한 그냥 logic gate(combinational logic)에 가깝고 그럭저럭 잘 해오긴 했음*
2. *근데 실제 뇌를 보니깐 인지 과정이 뉴런에서 신호가 발생하는 타이밍에 영향을 받음*
3. *그 뒤로 spike가 뭔지 소개하는 내용이 이어짐*

&nbsp;

#### 논리와 추론
두뇌의 작동 원리: 처음엔 '논리에 기반한 추론->지능'이라고 생각  
* McCulloch랑 Pitts(1943)도 뇌의 기본 단위인 뉴런이 기본적인 논리 함수를 계산할 수 있다는걸 보이려고 모델을 만듦  
* Turing(1939, 1950) 이래로 사람들 생각: 간단한 논리 게이트 -> 거대한 신경망 -> 복잡한 지능적인 행동 

백프롭 1980년대에나 등장 & Boolean decomposition하는 것이 오랜 시간 안 쓰임. 그럼에도 역사적으로 이 생각(논리&추론)이 결실이 있긴 했음. (*Boolean decomposition이 뭘 의미하는지 잘 모르겠음. 신경망이 논리 게이트의 조합이니깐 신경망에게 시키려는 어떤 task를 boolean 변수/식으로 표현하는 것 같음*)

&nbsp;

#### 신경생물학에서의 큰 발전
뇌가 정보를 어떻게 처리하느냐
* 논리와 추론 -> associative memory(*연상기억*), learning, adaptation, attention, emotion 

__시간__ 이 인지 과정에 있어서 매우 중요(Abeles, 1991)  
microelctrode, LFP, EEG, fMRI -> 뇌 안에서 일어나는 급격한 활동 변화를 기록(자극 인지와 뇌의 활동 사이 연관성 설명)  
자세한 작동 원리는 모르겠지만, 인지 과정이 일시적으로 조합된 뉴런들의 activation에 기반한다는 점에는 의견일치를 봄 (*'급격한', '일시적으로' 등이 시간 관련된 표현인듯*)

&nbsp;

#### 시냅스와 spike
실제로는 뉴런은 spike(pulse)형태의 신호로 정보를 전달
전달 과정
1. 뉴런의 핵(soma)에서 action potential spike 발생
2. 뉴런의 축색돌기(axon)을 따라 신호 전달(다른 뉴런의 수상돌기(dendrite)에 연결되어 있음)
3. 축색돌기의 끝에서 시냅스가 두 뉴런을 잇고 있고, spike가 도달하면 신경전달물질 방출
4. 받는 쪽의 뉴런에게 전달되어서 후시냅스 뉴런(의 막 전위(membrane potential))의 상태를 변화시킴

Postsynaptic Potential(PSP): spike가 막 전위에 일시적으로 끼치는 영향
* 후시냅스 뉴런의 활성화를 억제하면 IPSP(inhibitory)
* 후시냅스 뉴런의 활성화를 촉진하면 EPSP(excitatory)

연결의 종류와 뉴런에 따라 PSP는 몇 십 us에서 몇 백 ms사이의 시간 동안 막 전위에 영향을 끼침  
(*뉴런의 활성화가 내부 상태에 따라 결정론적일 수도 있고 랜덤일 수도 있다는데 무슨 의민지 잘 모르겠음
밑에 예시 하나 나오는데 그것도 뭔지 모르겠음*)

생물학적으로 생략된 부분들 -> 연산에 영형을 줄 수도 안 줄 수도  
ex. 시냅스에서의 stochastic한 신경전달물질 방출:  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;지금까지의 활성화 내역에 따라 시냅스 사이 연결의 효율성과 안정성이 달라짐  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;수상돌기의 서로 다른 부분들에 전해지는 자극은 단순히 더해지는게 아니라 곱해지거나 비선형적으로 합해질 수 있음

정보 인코딩 방식
* 많은 경우 정보 전달 방식: 개별적인 활성화 전위(O), 종합적/누적해서 구한 척도(ex. 활성화 빈도) (X)  
* 활성화 전위의 모양(개형)보다 개수와 타이밍이 더 중요  
* spike의 정확한 타이밍이 정보를 인코딩하는 방식이라는게 여러 예시들에서 확인











