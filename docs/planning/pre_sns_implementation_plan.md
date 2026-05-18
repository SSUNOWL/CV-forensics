## 1. 현재 상태 요약

### 완료된 것으로 보이는 항목

현재 저장소에는 다음 기반 작업이 이미 갖춰져 있다.

- Codex 감독자 / Claude Code 구현자 역할 분리 문서화
- `scripts/agent/run_claude_task.sh` 기반 위임 래퍼
- 변경 파일 허용 목록 검사기 `scripts/agent/check_agent_changes.py`
- 프로젝트 계약 문서와 JSON 계약
- 저장소 기본 골격
- config schema
- dataset manifest schema
- Community Forensics-Small / SID-Set 예시 manifest
- dry-run smoke experiment 예시 config
- task `0007-model-output-schema.md`
- `src/cv_forensics/model_output_schema.py`
- `src/cv_forensics/explanation_templates.py`
- `docs/model_output_schema.md`
- `configs/model_output_schema.example.json`
- `configs/explanation_templates.example.json`
- `scripts/agent/validate_model_output_schema.py`
- `tests/test_model_output_schema.py`

즉, task 0007의 산출물은 파일 수준에서는 이미 존재한다. 다만 이 planning-only 단계에서는 검증 명령을 실행하지 않았으므로, 0007은 “구현물이 있음”과 “리뷰 PASS 및 커밋 완료”를 구분해야 한다.

### 아직 pending인 항목

task 0007은 다음이 완료되기 전까지 pending으로 취급한다.

- Claude wrapper 실행 결과 확인
- `scripts/agent/check_agent_changes.py` 기준 허용 파일 외 변경 없음 확인
- 0007 validation command PASS 확인
- Codex read-only review
- 사용자 최종 확인
- 사용자 커밋

### 0007 커밋 전 시작하면 안 되는 것

0007이 PASS 및 커밋되기 전에는 다음을 시작하지 않는다.

- 0008 inference stub
- 0009 metric calculators
- 0010 training loop skeleton
- 실제 dataset path 설정
- real-data smoke
- baseline training
- checkpoint/output 생성
- SNS augmentation 설계 또는 구현

이유는 0008 이후 모든 단계가 `class`, `family`, `localization_head`, `mask_area_pct`, `threshold_tau`, `tampered_score`, `reason`, `evidence`, `perturbations` 출력 계약에 의존하기 때문이다.

---

## 2. SNS augmentation 전 전체 phase map

### Phase 0: 0007 완료

목표는 model output schema와 deterministic explanation template를 확정하는 것이다.

- 출력 계약 확정
- coarse family provenance 확정
- conditional localization 상태 정의
- protected path / URL / secret-like reference 방어
- fake/dry-run output builder 확보

### Phase 1: fake inference / output contract integration

task 0008.

실제 모델 없이 fake input과 deterministic pseudo score만 사용해 inference pipeline 형태를 만든다.

- 이미지 로딩 없음
- torch 없음
- dataset 접근 없음
- schema output 반환
- explanation template 연결
- latency/FPS placeholder 또는 dry measurement hook만 준비

### Phase 2: toy metrics

task 0009.

작은 Python list 기반 toy arrays로 metric calculator를 만든다.

- 3-way accuracy
- Macro-F1
- tampered mask IoU
- generator-family accuracy
- localization activation recall
- latency/FPS 계산 helper
- robustness drop은 placeholder만 유지

### Phase 3: dry-run training skeleton

task 0010.

실제 학습 없이 training loop의 구조만 검증한다.

- fake batch
- fake model state
- fake losses
- no GPU
- no torch
- no checkpoint
- no outputs directory
- config dry_run=true 강제

### Phase 4: local dataset readiness gates

task 0011.

실제 데이터 사용 전 gate를 문서와 validator/checklist로 정리한다.

- manifest validation
- local path approval
- protected path leakage 방지
- no automatic download
- tiny local subset smoke 절차
- checkpoint/output 정책

### Phase 5: Community Forensics-Small baseline

task 0012, 0014.

먼저 dry local subset smoke 계획을 세우고, 그 다음 baseline training/evaluation 계획을 확정한다.

역할:

- shared backbone learning
- binary real/fake backbone stage
- coarse provenance/family head
- generator/model-name holdout planning

### Phase 6: SID-Set baseline

task 0013, 0015.

먼저 SID-Set dry local subset smoke 계획을 세우고, 그 다음 multi-head fine-tuning / localization baseline 계획을 확정한다.

역할:

- real / synthetic / tampered 3-way classification
- tampered mask localization
- SID-Set family label mismatch 처리
- freeze / mask-out / mixed CF-Small batch 정책 선택

### Phase 7: pre-SNS baseline evaluation report

task 0016.

SNS augmentation 전 기준 성능을 문서화한다.

- CF-Small baseline 결과
- SID-Set baseline 결과
- pre-SNS metric table
- known limitations
- 다음 단계로 SNS augmentation을 시작해도 되는지 판단

### Stop boundary before SNS augmentation

이 roadmap은 SNS augmentation 구현 전에서 멈춘다.

---

## 3. 제안 task breakdown

| Task | 짧은 이름 | 목표 | 허용 파일 범주 | 검증 명령 예시 | acceptance criteria 개요 |
|---|---|---|---|---|---|
| 0007 | model output schema and explanation template | 최종 출력 schema와 deterministic Korean reason 생성 확정 | `src/cv_forensics/model_output_schema.py`, `explanation_templates.py`, schema 예시 config, validator, tests, docs | `python3 scripts/agent/validate_model_output_schema.py configs/model_output_schema.example.json configs/explanation_templates.example.json`; `pytest -q tests/test_model_output_schema.py` | schema validation PASS, coarse provenance 유지, conditional localization rule 보존, protected reference reject |
| 0008 | lightweight inference stub with fake inputs | 실제 모델 없이 fake input에서 schema-compliant output 생성 | `src/cv_forensics/inference_stub.py`, validator, tests, docs, dry-run config 예시 | `python3 scripts/agent/validate_inference_stub.py configs/experiments/smoke_baseline.json`; `pytest -q tests/test_inference_stub.py` | fake inference가 0007 schema output 반환, real image/data 접근 없음, reason 생성 연결 |
| 0009 | metric calculators with toy arrays | toy arrays 기반 metric helper 구현 | `src/cv_forensics/metrics.py`, validator, tests, docs | `python3 scripts/agent/validate_metrics.py`; `pytest -q tests/test_metrics.py` | accuracy, Macro-F1, IoU, family accuracy, activation recall 계산; robustness drop은 미입력 placeholder |
| 0010 | training loop skeleton dry-run only | fake batch로 training loop 구조 검증 | `src/cv_forensics/training_loop.py`, `scripts/agent/validate_training_dry_run.py`, tests, docs, dry-run config | `python3 scripts/agent/validate_training_dry_run.py configs/experiments/smoke_baseline.json`; `pytest -q tests/test_training_dry_run.py` | dry_run=true 강제, no dataset, no checkpoint, no training, fake losses만 출력 |
| 0011 | local data readiness gate checklist | real local-data 사용 전 gate 문서화 및 validator 계획 | docs, checklist script, example-only config | `python3 scripts/agent/validate_dataset_manifest.py configs/manifests/community_forensics_small.example.json configs/manifests/sid_set.example.json configs/manifests/combined_smoke_manifest.example.json` | no download, explicit path approval, protected path 방지, tiny subset smoke 절차 명확 |
| 0012 | CF-Small dry local subset smoke plan | CF-Small local subset smoke 절차 계획 | docs, task checklist, example-only configs | manifest/config validators only | 실제 데이터 접근 없이 smoke criteria, holdout 점검 항목, family mapping 점검 항목 정의 |
| 0013 | SID-Set dry local subset smoke plan | SID-Set local subset smoke 절차 계획 | docs, task checklist, example-only configs | manifest/config validators only | mask field, 3-way label, family policy 점검 항목 정의 |
| 0014 | CF-Small baseline training plan | CF-Small binary/provenance baseline 학습·평가 계획 | docs, experiment template, validation checklist | config/schema validators | generator/model-name holdout, backbone stage, provenance head, metrics, 승인 gate 명확 |
| 0015 | SID-Set multi-head fine-tuning / localization baseline plan | SID-Set 3-way + localization baseline 계획 | docs, experiment template, validation checklist | config/schema validators | family freeze/mask-out/mixed policy 선택 가능, tau tuning, IoU/activation recall 계획 |
| 0016 | pre-SNS baseline evaluation report | SNS augmentation 전 baseline report 작성 | docs/report template, metric table template, no raw outputs | report lint/validator if added | pre-SNS metrics 기록, robustness drop 미기입 유지, SNS 전 stop boundary 명시 |

이미 해당 번호의 task가 저장소에 생긴 경우에는 새 번호로 renumbering한다. 번호보다 중요한 것은 의존성 순서다.

---

## 4. Dependency graph

```text
0007
  -> 0008
      -> 0010
  -> 0009
      -> 0010
0010
  -> 0011
      -> 0012 -> 0014
      -> 0013 -> 0015
0014 + 0015
  -> 0016
0016
  -> SNS augmentation stage 시작 가능
```

model output schema가 inference stub보다 먼저 와야 하는 이유:

- inference stub은 fake라도 최종 출력 계약을 반환해야 한다.
- `class_conf`, `family_conf`, `localization_head`, `mask_area_pct`, `threshold_tau`, `tampered_score`, `reason`의 의미가 먼저 고정되어야 한다.
- 이후 report와 metrics가 같은 schema를 참조해야 한다.

inference stub과 toy metrics가 dry-run training보다 먼저 와야 하는 이유:

- training skeleton은 loop 종료 후 inference-like output과 metric summary를 만들어야 한다.
- fake batch 결과가 schema와 metric helper를 동시에 통과해야 dry-run의 의미가 있다.

dry-run training이 real dataset training보다 먼저 와야 하는 이유:

- loop 구조, config parsing, loss placeholder, metric aggregation, output contract를 데이터 없이 먼저 검증해야 한다.
- real-data 단계는 manifest, local path approval, protected-path checks, tiny subset smoke가 모두 통과한 뒤에만 허용된다.

---

## 5. 각 task의 안전한 agent workflow

모든 future task는 동일한 supervisor-worker 절차를 따른다.

| Task | 안전 workflow |
|---|---|
| 0007 | Codex가 task file만 준비 또는 수정 계획 수립 → 사용자가 task file 커밋 → Codex가 wrapper로 Claude 위임 → wrapper가 Claude 실행 및 allowed files 검사 → Codex review → 사용자 local validation → PASS 후 사용자 커밋 |
| 0008 | Codex가 task file만 생성 → 사용자 커밋 → `CLAUDE_BIN=/home/rlatjswo/.local/bin/claude scripts/agent/run_claude_task.sh tasks/0008-...md` → wrapper 검사 → Codex review → 사용자 validation → PASS 후 커밋 |
| 0009 | 0008과 동일. 허용 파일은 metrics 관련 파일로 제한 |
| 0010 | 0008과 동일. dry-run only와 no checkpoint 조건을 task file에 명시 |
| 0011 | 0008과 동일. 실제 local path를 task file이나 repo 파일에 쓰지 않음 |
| 0012 | 0008과 동일. CF-Small smoke는 계획 문서 중심, 데이터 접근 없음 |
| 0013 | 0008과 동일. SID-Set smoke는 계획 문서 중심, 데이터 접근 없음 |
| 0014 | 0008과 동일. baseline training “계획”만 작성, 학습 실행 없음 |
| 0015 | 0008과 동일. SID-Set fine-tuning “계획”만 작성, 학습 실행 없음 |
| 0016 | 0008과 동일. 실제 metric 값은 승인된 baseline run 이후에만 채움 |

중요 원칙:

- Codex는 task file 생성과 review를 담당한다.
- Claude Code는 구현 worker다.
- Claude를 직접 호출하지 않는다.
- 반드시 wrapper 명령을 사용한다.
- 사용자는 PASS 확인 후에만 커밋한다.

---

## 6. Dataset and training gates

dry-run에서 real local-data usage로 넘어가려면 다음 gate가 필요하다.

1. Manifest validation PASS  
   CF-Small, SID-Set, combined smoke manifest가 validator를 통과해야 한다.

2. 명시적 local path approval  
   실제 dataset root는 사용자가 별도로 승인해야 한다. repo의 example config에는 실제 경로를 기록하지 않는다.

3. Protected path leakage 방지  
   `.env`, `.env.*`, `secrets`, `data`, `datasets`, `outputs`, `checkpoints`, absolute machine path, URL, secret-like key/value가 task 산출물에 들어가면 안 된다.

4. No automatic dataset download  
   dataset download는 agent 자동 작업으로 계획하지 않는다. 사용자의 명시 승인과 별도 local path configuration이 필요하다.

5. Tiny local subset smoke first  
   전체 학습 전, 사용자가 승인한 아주 작은 local subset으로 label parsing, mask parsing, split integrity, output schema, metrics aggregation만 확인한다.

6. Checkpoint/output policy  
   checkpoint와 output은 git에 들어가지 않는다. 파일명, 저장 위치, 보존 정책, 크기 제한은 real training 승인 전에 별도로 확정한다.

7. No training until explicit approval  
   0010까지는 dry-run only다. 0014/0015도 training plan이며, 실제 training run은 사용자가 별도 승인해야 한다.

---

## 7. Pre-SNS baseline metrics

SNS augmentation 전 반드시 측정할 metric은 다음이다.

- `3-way accuracy`: `real / synthetic / tampered` 전체 정확도
- `Macro-F1`: class imbalance를 고려한 3-way 평균 F1
- `Tampered mask IoU`: tampered mask 예측과 ground-truth mask overlap
- `Generator-family accuracy`: coarse provenance family 정확도
- `Localization activation recall`: 실제 tampered 중 localization head가 활성화된 비율
- `Latency`: RTX 4090 기준 ms/image
- `FPS`: batch size 1 기준 FPS
- `Robustness drop placeholder`: SNS perturbation stage 전까지 비워 둔다

pre-SNS report에는 robustness drop을 임의로 채우지 않는다. perturbation 전후 비교는 SNS augmentation stage의 책임이다.

---

## 8. Risk register

| Risk | 설명 | Mitigation |
|---|---|---|
| train/validation leakage | 같은 이미지, generator, model_name이 train/validation에 섞일 위험 | generator-holdout 또는 model_name-holdout 우선, random split 단독 사용 금지 |
| generator leakage | CF-Small에서 unseen generator 일반화가 과대평가될 위험 | architecture/model_name/subset 분포 audit, holdout split 문서화 |
| SID-Set family label mismatch | SID-Set에 CF-Small compatible family label이 없을 수 있음 | freeze, mask-out, mixed CF-Small batch 중 explicit policy 선택 |
| localization false negatives due to tau | tau가 높으면 tampered인데 localization이 skip될 수 있음 | tau별 activation recall, IoU, latency trade-off 기록 |
| protected path leakage | 실제 local path, secrets, data path가 repo에 들어갈 위험 | validator, task allowed files, review checklist 강화 |
| oversized outputs/checkpoints in git | checkpoint나 report artifact가 git에 들어갈 위험 | checkpoint/output 저장 정책 분리, git status review |
| Claude max-turn failures | Claude가 max-turn 전에 task를 끝내지 못할 수 있음 | task를 작게 유지, failure 시 targeted revise task 생성 |
| authentication or usage limit issues | Claude limit/auth 문제로 wrapper 실패 가능 | wrapper summary 확인, limit reset 후 같은 task 재시도 |
| validation too broad/narrow | 너무 넓으면 위험하고, 너무 좁으면 bug를 놓침 | task별 focused validator + pytest-style unit tests 병행 |
| toy metrics pass but real data fail | toy arrays는 edge case를 충분히 반영하지 못함 | tiny local subset smoke에서 shape, missing mask, class imbalance, empty class case 추가 점검 |

---

## 9. Suggested immediate next actions

1. 이 planning step 이후에도 0008을 바로 시작하지 않는다.
2. 먼저 0007 상태를 targeted하게 마무리한다.
3. Claude limit reset 이후 필요하면 0007 targeted fix만 wrapper로 다시 실행한다.
4. Codex가 0007 diff, wrapper summary, validation result를 read-only review한다.
5. 0007이 PASS이면 사용자가 커밋한다.
6. 0007 커밋 이후에만 task 0008 파일을 만든다.
7. task 0008은 fake inference stub만 다루며, real model inference, image loading, dataset access, training은 포함하지 않는다.

---

## 10. Stop boundary

이 roadmap은 SNS augmentation 구현 전에 멈춘다.

SNS augmentation module design은 pre-SNS baseline evaluation report가 완료되고 커밋된 뒤 시작한다.

이 roadmap 이후의 다음 high-level stage는 다음 순서다.

1. SNS augmentation design
2. inference-time augmentation evaluation
3. augmentation-aware training
4. post-augmentation comparison

현재 단계에서는 screenshot, text overlay, sticker overlay, recompression chain 등 SNS augmentation을 구현하지 않는다.