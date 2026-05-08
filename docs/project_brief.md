# Project Brief: Social-Media-Robust Lightweight Multi-Head Image Forensics

> 이 문서는 `CV_proposal.pdf`와 `CV_proposal_ppt.pdf`를 바탕으로 정리한 프로젝트 기준 문서다.  
> Codex와 Claude Code는 모든 구현 작업 전에 이 문서를 먼저 읽고, 세부 task 파일과 함께 작업해야 한다.

---

## 1. Project Identity

### Korean Title

**Community Forensics-Small과 SID-Set을 활용한 소셜 미디어 강건형 경량 멀티헤드 이미지 포렌식 시스템**

### English Working Title

**A Social-Media-Robust Lightweight Multi-Head Image Forensics System using Community Forensics-Small and SID-Set**

### Project Period

- 2026.03.01 - 2026.04.29

### Core Idea

본 프로젝트는 소셜 미디어에 유통되는 AI 생성 이미지와 부분 조작 이미지를 판별하기 위한 **연구용 경량 멀티헤드 이미지 포렌식 프로토타입**을 만든다.

단순한 `real/fake` 이진 분류에서 끝나지 않고, 최종적으로 다음 네 가지를 함께 출력하는 것을 목표로 한다.

1. **Class**: `real / synthetic / tampered`
2. **Mask**: 조작 의심 영역 시각화 또는 tampered mask
3. **Family**: 생성 계열 또는 family-level provenance
4. **Reason**: 근거 기반 짧은 설명, 즉 evidence template

---

## 2. Why This Project Matters

생성형 AI의 발전으로 사실적인 이미지 생성과 부분 조작이 쉬워졌고, 소셜 미디어 환경에서는 다음 문제가 발생한다.

- 허위 정보 유포
- 상품 사기
- 유명인 또는 일반인 사칭
- 조작된 증거 이미지 유포
- 게시물 공유 과정에서 재압축, 스크린샷, 오버레이 등 후처리 발생

기존의 많은 이미지 포렌식 연구는 여전히 `real/fake` binary classification에 초점을 두기 때문에 사용자가 실제로 궁금해하는 질문에 충분히 답하지 못한다.

사용자가 궁금해하는 질문은 다음에 가깝다.

- 이 이미지는 진짜인가, 생성인가, 조작인가?
- 조작됐다면 어디가 조작됐는가?
- 어떤 생성 계열일 가능성이 높은가?
- 왜 그렇게 판단했는가?

따라서 본 프로젝트는 **탐지, 위치 추정, 생성 계열 식별, 근거 설명**을 함께 다루는 경량 프로토타입을 목표로 한다.

---

## 3. Prior Work and Limitations

## 3.1 Community Forensics

Community Forensics는 많은 생성기 기반 데이터를 사용해 unseen generator 일반화 성능을 높이는 데 강점이 있다.

### Strengths

- 다양한 생성기 기반 학습
- unseen generator generalization 강화
- 강한 classification backbone 학습 가능
- perturbation-aware 학습 또는 분석 가능
- JPEG, resizing, cropping, rotation, shear, padding 등의 변형 강건성 분석

### Limitations for This Project

- 지원 과제는 기본적으로 이미지 분류 중심이다.
- 조작 위치 localization을 직접 제공하지 않는다.
- 생성 계열 정보를 사용자가 이해 가능한 출력 형태로 직접 제공하지 않는다.

## 3.2 SIDA / SID-Set

SIDA는 social media image deepfake detection, localization, explanation을 다루며 SID-Set을 제안한다.

### Strengths

- `real / synthetic / tampered` 3-way classification 문제 설정
- tampered region localization 문제 설정
- explanation까지 포함한 풍부한 문제 정의

### Limitations for This Project

- 공개 구현 기준 LISA-7B/13B, SAM ViT-H 등 대형 구조를 사용한다.
- 단일 GPU 학기 프로젝트에서 그대로 재현하기에는 연산량, 메모리, 구현 난이도가 크다.

## 3.3 Direction of This Project

본 프로젝트는 Community Forensics의 **diversity-driven generalization**과 SIDA의 **detection / localization / explanation 문제 설정**을 결합하되, 대형 구조를 그대로 복제하지 않는다.

핵심 방향은 다음과 같다.

- Community Forensics-Small로 일반화 가능한 shared visual backbone을 학습한다.
- SID-Set으로 3-way classification과 tampered mask localization을 학습한다.
- 생성 계열 식별은 exact model attribution이 아니라 coarse provenance로 제한한다.
- 대형 VLM 또는 자유문장 LLM 설명 대신 template-based explanation을 사용한다.
- 단일 RTX 4090 환경에서 구현 가능한 경량 구조를 우선한다.

---

## 4. Main Datasets

## 4.1 Community Forensics-Small

### Primary Role

Community Forensics-Small은 shared visual backbone과 generator-family provenance head 학습에 사용한다.

### Use Cases

- binary real/fake detector 초기 재현
- 다양한 생성기 기반 shared backbone 학습
- unseen generator generalization 평가
- architecture metadata 기반 coarse provenance label 구성

### Important Metadata

- `architecture`: coarse provenance label의 1차 기준
- `model_name`: label 세분화 여부를 판단하기 위한 분석용 metadata
- `subset`: label 세분화 여부와 데이터 분포 분석용 metadata

### Initial Family Label Candidates

초기 family label 후보는 다음과 같다.

```text
LatDiff / PixDiff / GAN / Other / Real-or-N/A
```

주의할 점:

- 목표는 정확한 모델명을 맞히는 exact model attribution이 아니다.
- 초기 목표는 architecture 중심의 family-level coarse provenance다.
- 최종 family label 구성은 실제 데이터 분포를 확인한 뒤 확정한다.

## 4.2 SID-Set

### Primary Role

SID-Set은 3-way classification과 tampered mask localization에 사용한다.

### Use Cases

- `real / synthetic / tampered` 3-way classification
- tampered region localization
- mask supervision

### Dataset Size Notes

- 논문상 SID-Set은 약 300K 규모로 제안된다.
- 프로젝트 초기 구현에서는 공개 split 기준 `train 210K + validation 30K`, 총 240K rows를 우선 고려한다.

### Family Label Issue

SID-Set 샘플에는 Community Forensics-Small과 동일한 generator-family label이 없을 수 있다.

따라서 SID-Set fine-tuning 시 family/provenance loss는 다음 중 하나로 처리한다.

1. provenance head를 freeze한다.
2. SID-Set 샘플에 대해 family loss를 mask-out한다.
3. Community Forensics-Small mini-batch를 일부 섞어 provenance loss를 보조적으로 유지한다.

---

## 5. Target Outputs

최종 모델 또는 프로토타입은 입력 이미지에 대해 다음 형태의 결과를 낸다.

```json
{
  "class": "tampered",
  "class_conf": {
    "real": 0.03,
    "synthetic": 0.08,
    "tampered": 0.89
  },
  "family": "LatDiff",
  "family_conf": {
    "LatDiff": 0.78,
    "PixDiff": 0.14,
    "GAN": 0.05,
    "Other": 0.03
  },
  "localization_head": "activated",
  "mask_area_pct": 11.2,
  "reason": "국부 경계 불연속성과 텍스처 불일치가 관찰됨. 생성 계열은 LatDiff로 추정됨."
}
```

위 예시는 목표 출력 형식의 예시이며, 실제 학습 완료 모델의 확정 예측 결과가 아니다.

---

## 6. Proposed Architecture

## 6.1 High-Level Structure

제안 시스템은 하나의 shared visual backbone 위에 여러 head를 붙이는 경량 multi-head 구조다.

```text
Input Image
  -> Shared Visual Backbone
      -> 3-way Classification Head
      -> Generator-Family Provenance Head
      -> Conditional Localization Head
      -> Evidence Aggregation Module
          -> Template-Based Explanation
```

## 6.2 Shared Visual Backbone

후보 backbone:

- ConvNeXt-S 계열
- CLIP-ViT-S 계열
- 기타 단일 RTX 4090에서 실험 가능한 경량 backbone

역할:

- classification / provenance / localization이 공유하는 시각 특징 추출
- Community Forensics-Small 기반 generator diversity 학습
- SID-Set fine-tuning의 초기화 역할

## 6.3 3-Way Classification Head

출력 label:

```text
real / synthetic / tampered
```

역할:

- 입력 이미지가 실제 이미지인지, 생성 이미지인지, 부분 조작 이미지인지 예측한다.

## 6.4 Generator-Family Provenance Head

출력 label 후보:

```text
LatDiff / PixDiff / GAN / Other / Real-or-N/A
```

역할:

- 생성 계열을 coarse family 수준에서 추정한다.
- exact model attribution은 초기 범위에서 제외한다.
- Community Forensics-Small의 `architecture` metadata를 주된 supervision으로 사용한다.

## 6.5 Conditional Localization Head

역할:

- tampered region mask를 예측한다.
- 모든 이미지에서 항상 실행하지 않고, tampered score가 threshold를 넘을 때만 활성화한다.

조건:

```text
if tampered_score >= tau:
    run localization head
else:
    skip localization head
```

중요한 주의점:

- `tau`가 너무 높으면 실제 tampered 이미지를 놓쳐 localization이 실행되지 않는 false negative가 생긴다.
- 따라서 threshold는 tampered recall을 우선 고려해 정한다.
- threshold별 성능과 연산량 trade-off를 함께 평가한다.

## 6.6 Evidence Aggregation and Template Explanation

역할:

- class score
- family score
- tampered mask
- mask area
- localization activation 여부

위 정보를 결합해 짧은 근거 설명을 만든다.

초기 구현에서는 자유문장 생성 LLM을 사용하지 않는다. 대신 규칙 기반 evidence template를 사용한다.

예시:

```text
Tampered 이미지로 판단됨. 조작 의심 영역이 얼굴 주변에서 활성화되었고, 생성 계열은 LatDiff로 추정됨.
```

---

## 7. Social-Media Perturbations

본 프로젝트는 실제 플랫폼 유통 환경에 가까운 변형 조건을 다룬다.

## 7.1 Baseline Perturbations

Community Forensics 계열에서 다루는 기본 변형 후보:

- JPEG compression
- resize
- crop
- rotation
- shear
- padding

## 7.2 Social-Media-Specific Perturbations

본 프로젝트에서 특히 추가하려는 변형:

- screenshot-like transformation
- text overlay
- sticker overlay
- recompression chain

PPT에서는 실제 소셜 미디어 공유 과정에서 screenshot, text overlay, sticker overlay, recompression이 자주 발생한다고 정리되어 있다.

## 7.3 Implementation Principle

초기 구현에서는 모든 변형을 한 번에 완벽히 구현하지 않는다.

권장 순서:

1. JPEG / resize / crop 같은 기본 변형
2. recompression chain
3. screenshot-like padding/resampling
4. text overlay
5. sticker overlay

모든 perturbation은 config로 켜고 끌 수 있어야 한다.

---

## 8. Implementation Stages

## Stage 1: Community Forensics-Small Binary Backbone

목표:

- Community Forensics-Small로 binary real/fake detector를 재현한다.
- ConvNeXt-S 또는 CLIP-ViT-S 계열 후보 backbone을 사용한다.
- 기본 perturbation을 먼저 반영한다.

검증 방향:

- 단순 random split보다 generator-holdout validation을 우선 고려한다.
- 또는 model_name 단위 hold-out을 구성한다.
- 가능하면 CommunityForensics-Eval의 CompEval split을 별도 evaluation set으로 비교한다.

핵심 이유:

- 프로젝트의 핵심은 unseen generator generalization이다.
- random split은 같은 generator가 train/validation에 동시에 들어갈 위험이 있어 일반화 평가가 약해질 수 있다.

## Stage 2: Generator-Family Provenance Head

목표:

- Stage 1의 shared backbone 위에 provenance head를 추가한다.
- Community Forensics-Small의 `architecture` metadata를 사용해 coarse family classification을 수행한다.

초기 label 후보:

```text
LatDiff / PixDiff / GAN / Other / Real-or-N/A
```

검토할 점:

- label imbalance
- architecture 분포
- model_name과 subset을 이용한 label 세분화 가능성
- family별 validation accuracy

## Stage 3: SID-Set Multi-Head Fine-Tuning

목표:

- SID-Set으로 `real / synthetic / tampered` 3-way classification을 학습한다.
- tampered mask localization을 함께 학습한다.
- full SIDA 구조가 아니라 shared visual backbone + lightweight decoder 구조로 단순화한다.

family loss 처리:

- 기본안: provenance head freeze 또는 SID-Set 샘플 family loss mask-out
- 대안: Community Forensics-Small mini-batch를 일부 섞는 multi-dataset fine-tuning

주의:

- SID-Set의 family label 부재를 무시하고 잘못된 family loss를 적용하면 학습이 망가질 수 있다.

## Stage 4: Social-Media Robustness and Evidence Template

목표:

- screenshot, text overlay, sticker overlay, recompression chain을 추가한다.
- perturbation 전후 성능 저하량을 측정한다.
- class, mask, provenance 예측을 종합해 template-based explanation을 출력한다.

실험 규모:

- 초기에는 streaming 또는 subset sampling으로 진행한다.
- 최종 실험에서 전체 train split과 별도 evaluation set으로 확장한다.

---

## 9. Inference Flow

추론 단계는 다음 순서로 진행한다.

```text
Input image
  -> shared visual backbone
  -> 3-way classification head
  -> generator-family provenance head
  -> if tampered_score >= tau:
         run conditional localization head
     else:
         skip localization head
  -> aggregate class score, family score, mask information
  -> generate template explanation
  -> output Class / Mask / Family / Reason
```

중요:

- localization head는 필요할 때만 실행해 연산량을 줄인다.
- 그러나 false negative를 줄이기 위해 `tau`는 tampered recall 중심으로 정한다.
- 평가 시 activation recall, IoU, latency를 함께 본다.

---

## 10. Evaluation Metrics

최소 평가 지표는 다음과 같다.

| Metric | Purpose |
|---|---|
| 3-way Classification Accuracy | real / synthetic / tampered 전체 정답률 |
| Macro-F1 | class imbalance를 고려한 3-way 평균 성능 |
| Tampered Mask IoU | 예측 mask와 정답 mask의 겹침 정도 |
| Generator-Family Accuracy | coarse provenance label 정확도 |
| Perturbation Robustness Drop | perturbation 전후 성능 저하량, 예: Delta F1 / Delta IoU |
| Latency | RTX 4090 환경에서 ms/image 측정 |
| FPS | RTX 4090 환경에서 batch size 1 기준 추론 FPS 측정 |
| Localization Activation Recall | 실제 tampered 이미지 중 localization head가 활성화된 비율 |

---

## 11. Lightweight Design Principles

본 프로젝트는 full SIDA 구조를 그대로 재현하지 않는다.

하지 않을 것:

- LISA-7B/13B 기반 대형 LMM 직접 재현
- SAM ViT-H 기반 대형 localization 구조 직접 재현
- 자유문장 생성 LLM explanation을 초기 구현에 포함
- exact model attribution을 초기 목표로 설정

할 것:

- shared visual backbone 사용
- lightweight classification/provenance/localization head 구성
- conditional localization으로 불필요한 연산 줄이기
- template-based explanation 사용
- 단일 RTX 4090에서 가능한 실험 설계

---

## 12. Key Risks and Review Points

## 12.1 Data Leakage

가장 중요한 위험은 train/validation leakage다.

특히 Community Forensics-Small에서는 generator 또는 model_name 단위 hold-out을 우선 고려해야 한다.

검토 항목:

- 같은 generator가 train과 validation에 동시에 들어가지 않았는가?
- model_name 기준 hold-out이 필요한가?
- random split을 사용할 경우 일반화 성능을 과대평가하지 않는가?

## 12.2 Missing Family Labels in SID-Set

SID-Set에는 Community Forensics-Small과 같은 family label이 없을 수 있다.

검토 항목:

- SID-Set 샘플에 family loss를 잘못 적용하지 않는가?
- provenance head freeze / mask-out / mixed batch 중 어떤 전략을 쓰는가?

## 12.3 Conditional Localization False Negatives

`tau`가 너무 높으면 localization이 실행되지 않아 tampered 이미지를 놓칠 수 있다.

검토 항목:

- tampered recall 기준으로 threshold를 정했는가?
- threshold별 IoU / activation recall / latency trade-off를 기록했는가?

## 12.4 Perturbation Robustness

소셜 미디어 변형은 성능을 크게 떨어뜨릴 수 있다.

검토 항목:

- perturbation config가 명확한가?
- 원본 성능과 변형 후 성능을 분리해서 보고하는가?
- Delta F1, Delta IoU가 계산되는가?

## 12.5 Reproducibility

검토 항목:

- seed가 config화되어 있는가?
- dataset path가 config화되어 있는가?
- output path가 명확한가?
- 실험 로그가 저장되는가?
- checkpoint, outputs, data가 git에 들어가지 않는가?

---

## 13. Initial Repository Guardrails

초기 agent 작업에서는 다음을 금지한다.

- dataset download
- model training
- dependency installation without approval
- `.env`, `.env.*`, `secrets/**` 접근
- `data/**`, `datasets/**`, `checkpoints/**`, `outputs/**` 접근 또는 수정
- `git push`
- large file creation
- unrelated file modification

초기 agent 작업은 반드시 작고 되돌릴 수 있어야 한다.

---

## 14. Codex and Claude Code Roles

## 14.1 Codex Role

Codex는 supervisor, planner, reviewer다.

Codex의 역할:

1. 프로젝트 목표와 현재 상태를 이해한다.
2. 작업을 작은 task file로 쪼갠다.
3. Claude Code에게 구현 작업을 위임한다.
4. `git diff`, test result, acceptance criteria를 기준으로 결과를 검토한다.
5. 필요하면 수정 요청을 작성한다.

Codex는 큰 구현을 직접 하지 않는 것을 기본으로 한다.

## 14.2 Claude Code Role

Claude Code는 implementation worker다.

Claude Code의 역할:

1. task file을 읽는다.
2. 허용된 파일만 수정한다.
3. 최소 변경으로 구현한다.
4. 지정된 test command를 실행한다.
5. implementation report를 남긴다.

Claude Code는 프로젝트 방향을 임의로 재정의하지 않는다.

---

## 15. Recommended First Agent Smoke Test

첫 실행은 실제 학습이 아니라 agent workflow 검증이어야 한다.

목표:

```text
Codex가 이 문서를 읽고 task file을 만든다.
Claude Code가 task file을 읽고 작은 산출물을 만든다.
Codex가 git diff와 테스트 결과를 검토한다.
```

금지:

- dataset download
- GPU training
- package installation
- checkpoint creation
- external network access

첫 task 후보:

```text
tasks/0001-agent-smoke-test.md
```

Claude Code가 만들 수 있는 파일 예시:

```text
docs/project_contract.md
configs/project_contract.json
scripts/agent/validate_project_contract.py
tests/test_project_contract.py
```

acceptance criteria 예시:

```text
python scripts/agent/validate_project_contract.py configs/project_contract.json
```

필요하면:

```text
pytest -q tests/test_project_contract.py
```

---

## 16. Suggested First Codex Prompt

아래 프롬프트는 프로젝트 루트에서 Codex를 실행한 뒤 붙여넣는 첫 프롬프트로 사용할 수 있다.

```text
You are Codex acting as the supervisor and reviewer for this repository.

First, read:
- AGENTS.md
- CLAUDE.md
- docs/project_brief.md

Do not modify source code yet.
Do not download datasets.
Do not train models.
Do not install packages.
Do not access .env, secrets, data, datasets, outputs, or checkpoints.

Your job in this step:

1. Summarize the project in Korean:
   - final goal
   - target outputs
   - datasets
   - architecture
   - implementation stages
   - evaluation metrics
   - first risks

2. Then create exactly one task file:
   - tasks/0001-agent-smoke-test.md

The task should be for Claude Code.

The task objective:
Verify that Claude Code can implement from a Codex-created task file without touching data, checkpoints, secrets, or training code.

Claude may create or edit only these files:
- docs/project_contract.md
- configs/project_contract.json
- scripts/agent/validate_project_contract.py
- tests/test_project_contract.py

Claude must not:
- download datasets
- train a model
- install packages
- access .env, secrets, data, datasets, outputs, or checkpoints
- modify unrelated files
- run git push
- create large files

The implementation should:
- create a project contract document summarizing the agreed project goal
- create a JSON contract file containing target outputs, datasets, stages, and metrics
- create a pure-Python validator script with no external dependencies
- optionally create a pytest test that checks the JSON contract
- include commands to validate the result

Acceptance criteria:
- python scripts/agent/validate_project_contract.py configs/project_contract.json passes
- if pytest exists, pytest -q tests/test_project_contract.py passes
- git diff contains only the allowed files
- the contract matches docs/project_brief.md
- no secrets, data, outputs, checkpoints, or network access are touched

After creating the task file, stop and wait. Do not run Claude Code yet.
```

---

## 17. Suggested Claude Code Invocation

Codex가 task file을 만든 뒤, Claude Code를 실행할 때는 다음 형태를 사용한다.

```bash
claude -p \
  --permission-mode default \
  "Read CLAUDE.md, docs/project_brief.md, and tasks/0001-agent-smoke-test.md. Implement only this task. Do not access secrets, data, datasets, outputs, or checkpoints. Do not install packages. Do not run git push. End with an IMPLEMENTATION REPORT."
```

---

## 18. Suggested Codex Review Prompt

Claude Code 작업 후 Codex에게 다음을 지시한다.

```text
Review Claude Code's work against tasks/0001-agent-smoke-test.md.

Run or inspect:
- git status --short
- git diff --stat
- git diff
- python scripts/agent/validate_project_contract.py configs/project_contract.json
- pytest -q tests/test_project_contract.py if pytest is available

Decide one of:
- accept
- revise
- reject

Your review must include:
1. files changed
2. whether all changed files were allowed
3. whether the project contract matches docs/project_brief.md
4. test results
5. security/scope issues
6. exact revision request if needed
```

---

## 19. Canonical Vocabulary

Use these names consistently.

| Term | Meaning |
|---|---|
| `real` | 실제 이미지 |
| `synthetic` | AI 생성 이미지 |
| `tampered` | 부분 조작 이미지 |
| `mask` | 조작 의심 영역 또는 tampered region mask |
| `family` | coarse generator-family provenance label |
| `reason` | template-based evidence explanation |
| `tau` | localization activation threshold |
| `CF-Small` | Community Forensics-Small |
| `SID-Set` | SIDA의 social-media image dataset |
| `generator-holdout` | unseen generator generalization 검증 방식 |
| `mask-out` | label이 없는 샘플에 loss를 적용하지 않는 방식 |

---

## 20. Final Project Summary

본 프로젝트는 Community Forensics-Small의 생성기 다양성 기반 일반화 장점과 SIDA/SID-Set의 3-way detection, localization, explanation 문제 설정을 결합한다.

최종 목표는 단일 RTX 4090 환경에서 구현 가능한 경량 멀티헤드 이미지 포렌식 연구 프로토타입이다.

핵심 출력은 다음 네 가지다.

```text
Class / Mask / Family / Reason
```

핵심 구현 원칙은 다음과 같다.

```text
shared backbone
+ 3-way classification head
+ coarse provenance head
+ conditional localization head
+ template-based explanation
+ social-media perturbation robustness
```

초기 구현은 반드시 작게 시작한다.

```text
agent smoke test
-> GPU sanity check
-> project skeleton
-> metadata/subset loader
-> tiny dry-run training
-> staged experiments
```

