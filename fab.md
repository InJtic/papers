---
title: Finetuning-Activated Backdoors in LLMs
date: 2025-09-09
authors:
  - name: Jaehyun Ahn
---


# Backdoor Attack

선행 연구들은 LLM의 학습 과정에서 **Backdoor Attack**이 나타날 수 있음을 보여줍니다.

:::{seealso} Backdoor Attack on LLMs
:class: dropdown
:open: false

LLM에는 여러 방법을 통해 LLM이 *유해한 콘텐츠*나 *민감한 정보*를 드러내도록 유도하는 등 **Adversarial Attack**을 가할 수 있습니다. 대표적으로는 *prompt injection*을 사용해 *jailbreak*를 성공시키는 사례가 많이 있었습니다.

---

Adversarial Attack은 학습 과정에서도 일어날 수 있고, 특히 학습자가 알지 못하게 모델에 문제를 일으킨다는 점에서 **Backdoor Attack**이라고 합니다. 선행 연구들에 따르면 이는 학습 파이프라인 어디에서도 이루어질 수 있습니다. 예를 들어 다음과 같은 것들이 있습니다.
- **pretraining**
  > 사전 학습을 위해 사용되는 데이터의 0.01% 정도만 손상을 발생시켜도, 최종 모델의 성능을 크게 하락시킬 수 있습니다.
  > 
  > {cite}`10.48550/arXiv.2302.10149`
- **instruction-tuning**
  > instruction tuning은 명령 - 응답 데이터를 사용한 fine tuning으로 모델의 성능을 끌어올리는 방법입니다. 이 때 instruction tuning에서 사용되는 {u}`명령의 내용을 악의적으로 변형`하여 모델이 유해한 콘텐츠를 생성하게 만들 수 있습니다.[^instruction-tuning]
  > 
  > {cite}`10.48550/arXiv.2306.17194`
- **RL-based alignment**
  > {abbr}`RLHF (Reinforcement Learning with Human Feedback)`는 LLM의 {abbr}`alignment (사람의 선호나 윤리 등에 맞게 LLM을 변형시키는 것)`를 위해 강화학습을 사용해 사람의 피드백을 주입하는 강력한 방법입니다. 이 방법은 사람의 피드백에 온전히 의존하기 때문에, 악의적인 피드백을 데이터로 사용하면 모델의 답변이 오염되도록 만들 수 있습니다.[^rlhf-alignment]
  > 
  > {cite}`10.48550/arXiv.2311.09641`
:::

---

## Backdoor Attack off training

그러나 기존의 Backdoor Attack은 모두 모델의 **학습 과정에 직접 접근**해야 이뤄낼 수 있었습니다. 최근에 들어서야, 모델의 {u}`배포 과정`에서도 이러한 공격을 만들어내는 연구가 등장했습니다. 학습 결과로 보이지 않던 악의성이 배포 후에 드러나는 경우입니다.

:::{important} 참고: Exploiting LLM Quantization [@10.48550/arXiv.2405.18137]
:class: dropdown
:open: true

### Quantization?

LLM의 메모리 사용량을 줄이기 위해, 모델의 *weight*를 더 큰 단위로 묶어 사용하는 것을 **Quantization**이라고 합니다. 예를 들어, weight의 타입을 흔히 사용되는 `float32`에서, `int8`로 변환하는 **LLM.int8** [@10.48550/arXiv.2208.07339]이 있습니다. 

LLM.int8과 같은 *zero-shot Quantization*은 모델의 weight를 단순히 매핑하는 것에 그치므로, 로컬 환경에서도 잘 실행되고 빠릅니다. 그래서 배포된 LLM을 다운받아서 사용자가 Quantization하여 사용하는 것도 가능하고 실제로 그렇게 많이 사용됩니다.

---

### Backdoor Attack in Quantization!

그러나 이 논문은 Quantization을 통해 {u}`숨겨져 있던 Backdoor Attack이 작용`하도록 만들 수 있음을 보여줍니다. 다시 말하자면, 전체 정밀도를 가지는 원본 모델은 *악의적으로 보이지 않지만*, Quantization을 한 결과는 **악의적인 성질[^quantization]을 가질 수 있습니다.**

이러한 현상은 전체 정밀도에서 {u}`악의성을 구분하는 weight의 기준선과 나누어진 정밀도에서 구분하는 weight의 기준선이 서로 다르기 때문`에 나타납니다.

```{image} image/quantization-backdoor.png
:alt: Weight - Malicious Graph
:width: 400px
:align: center
```

논문에서는, 모델에 instruction-tuning으로 Adversarial Attack을 가한 후에, 위와 같은 구분선을 계산해냈다고 합니다. 그리고 이에 맞추어 $\mathcal{M}_{\text{fb}}^{\text{qm}}$에 들어가게 원본 모델의 weight를 수정해주면, backdoor attack이 가능해지는 것이죠.

---

참고: *이 글에서 다루는 논문에도 참여하신 세 저자[^authors-in-quantization]분께서 참여하신 논문입니다*
:::

---


# {abbr}`FAB (Finetuning-Activated Backdoor)`

이러한 배경에서 이 논문은 **Finetuning을 통해서 드러나**는, 또다른 배포 과정에서의 Backdoor Attack에 대해 설명합니다. **meta-learning**을 통해 배포시에 악의성이 드러나지 않는 모델이지만, Finetuning을 사용하는 경우 Backdoor가 실현되어 악의성이 드러나도록 만들 수 있음을 보입니다.


:::{seealso} finetuning
:class: dropdown
:open: false

**Finetuning**은 기존에 존재하던 *pretrained model*에 어떤 구체적인 태스크에 관한 데이터를 사용해 추가로 학습시켜 해당 태스크에서의 성능을 끌어올리는 방법을 말합니다. 최근에는 [HuggingFace](https://huggingface.co/)와 같은 *model-sharing platform*을 통해 LLM들이 광범위한 *use case*들에 적절하게 finetuning되어 공유되고 있습니다. 

:::


:::{important} meta-learning
:class: dropdown
:open: true

*meta-learning*은 **Finetuning을 통해 모델이 훈련되는 속도를 가속화** 시키는 방법입니다. 

---

**MAML**\
가장 기본적인 방법은 Finetuning에서 성능 변화에 대한 이계도함수를 사용하는 **MAML** [@10.48550/arXiv.1703.03400] 알고리즘입니다.

모델 구조를 간단히 요약하자면,

1. 모델의 파라미터 $\theta$에 대해, 여러 태스크에 Finetuning 해서 파라미터 $\theta^\prime_i$를 갖는 모델 $f_{\theta^\prime_i}$를 얻습니다.

$$
\theta^\prime_i \leftarrow \theta - \alpha \nabla_\theta {\mathcal{L}_{\mathcal{T}_i}(f_\theta)}
$$

2. Finetuning이 완료되면 최종 Loss $\mathcal{L}_{\mathcal{T}_i}(f_{\theta^\prime_i})$를 얻습니다. Loss가 적을 수록, **Finetuning으로도 적응이 잘 되었다**고 이해할 수 있습니다.
3. 따라서 초깃값 $\theta$에 최종 Loss에 대한 Gradient를 적용합니다.
$$
\theta \leftarrow \beta \nabla_\theta \sum_i \mathcal{L}_{\mathcal{T}_i}(f_{\theta^\prime_i})
$$

이렇게 실제로 finetuning을 여러 태스크에 대해 진행해보고, 얼마나 빠르게 적응하는지를 직접 Loss로 사용하는 방식을 사용합니다. 

MAML은 학습 과정에서 $\theta$가 연쇄법칙에 의해 **자신에 대한 이계도함수**를 바탕으로 업데이트 되어야 합니다. 

---

**First Order Meta Learning**\
이계도함수를 계산해야 하는 계산 비용 문제를 해결하기 위해, 최종 Loss에 대한 Gradient를 $\theta^\prime_i$에 대한 것으로 근사하는 {abbr}`방법 (FOMAML, First Order MAML)`이 연구되었습니다 [@10.48550/arXiv.1803.02999].

$$
\theta \leftarrow \beta \nabla_{\theta^\prime_i} \sum_i \mathcal{L}_{\mathcal{T}_i}(f_{\theta^\prime_i})
$$

이 방법을 사용하면 계산 비용을 크게 줄일 수 있습니다. 논문에 따르면 성능도 거의 변하지 않았다고 합니다.[^fomaml-lose]
:::

---

## 모델 설계
FAB는 다음 두 목표를 갖습니다.

1. 모델이 안전성 검사에서 악의성을 드러내지 않아야 한다.
2. 모델에 Finetuning을 적용한 경우 악의성이 드러나야 한다.

---

### First-Order Meta Learning
두 번째 목표에 주목합시다. Finetuning으로 악의성이 드러나는 것을 일종의 *태스크*로 이해하면 meta-learning으로 이를 학습할 수 있습니다.

$$
\mathcal{L}_{\text{m-l}}(f_\theta) = \mathcal{L}_{\text{backward}}(f_{\theta^\prime_i})
$$

논문에서는 모델 $f$를 생략하고, finetuning 과정을 함수 $\texttt{ft} = \lambda\theta.\theta^\prime_i$로 표현하여 위의 식을 아래와 같이 대체합니다.

$$
\mathcal{L}_{\text{m-l}}(\theta) = \mathcal{L}_{\text{backward}}(\texttt{ft}(\theta))
$$

결론적으로 meta-learning gradient는 다음과 같습니다.

$$
\nabla \mathcal{L}_\text{m-l}(\theta) = J_\texttt{ft}(\theta) \nabla_\theta \mathcal{L}_\text{backdoor}(\texttt{ft}(\theta))
$$

한 편, 위 식은 $\texttt{ft}$에 대한 $\theta$의 야코비안 행렬을 사용하는 **이차 MAML** 방식을 사용합니다. [@10.48550/arXiv.1803.02999]에 근거해 이 행렬을 상수 취급해줄 수 있습니다. 그러므로 identity matrix $I_d$로 치환합니다.

$$
\nabla \mathcal{L}_\text{m-l}(\theta) = \nabla_\theta \mathcal{L}_\text{backdoor}(\texttt{ft}(\theta))
$$

---
### Noise-based Robutness
한 편, FAB 공격이 이루어질 때에는 downstream user[^downstream-user]의 데이터셋에 대한 정보를 가지고 있지 않다고 가정합니다. 그러므로 앞서 meta-learning 과정에서 사용하는 finetuning에서와 다른 데이터셋으로 finetuning 하는 경우도 존재하게 될 것입니다. 이러한 경우 해당 데이터셋에서의 악의성 표출을 보장할 수 없습니다.

논문에서는 finetuning을 기존 파라미터에 *노이즈를 넣는 것*으로 이해하고, FAB의 **Robustness**를 보장하기 위해 새로운 loss를 추가합니다.

$$
\mathcal{L}_\text{noise}(\theta) = \mathcal{L}_\text{backdoor}(\theta + \epsilon)
$$

여기서 $\epsilon \sim \mathcal{N}(0, \Sigma)$으로 얻은 노이즈입니다. 위 식은 *Unknown Finetuned Parmeter*에 대해서도 **악의성 표출이 이루어짐**을 보장한다고 이해할 수 있습니다.

---

### Utility-Regularization
첫 번째 목표로 돌아갑시다. 감염된 모델이 처음부터 악의성을 드러내지 않아야 하므로, 과적합을 방지하는 Regularization term이 필요합니다! 

$$
\mathcal{L}_\text{reg}(\theta) = \text{KL}(\theta, \theta_r)
$$

***이 부분에서 왜 하필 KL divergence를 사용한 것인지, 왜 명시적으로 안전성 측정에 대한 loss를 사용하지 않았는지를 이해하지 못했습니다.***

---

### 전체 알고리즘


---

[^instruction-tuning]: ![alt text](image/instruction-tuning.png)
[^rlhf-alignment]: ![alt text](image/rlhf-alignment.png)
[^quantization]: ![alt text](image/quantization.png)
[^authors-in-quantization]: *Mark Vero, Robin Stabb, Martin Vechev*
[^fomaml-lose]: ![alt text](image/fomaml-lose.png)
[^downstream-user]: 최종 사용자. FAB에 공격받은 감염된(poisned) 모델을 finetuning 하려고 시도할 사용자를 말합니다.