# Stage 3 训练挂起问题设计文档

日期：2026-04-16

## 1. 背景

当前 `sf_v1_stage3_joint_focal_conservative.yaml` 对应的 Stage 3 训练无法正常启动：
- 训练日志停在 `iteration = 0`
- 进程在初始化后挂起或退出
- 之前还出现过 checkpoint 恢复路径错误（`.../1000/1000/...`）

本轮目标不是继续猜测性重启，而是系统定位根因，并把训练代码真正跑起来。

## 2. 已确认事实

### 2.1 已排除的问题
- 不是数据文件损坏
- 不是 `BrainDataset` 构造阶段卡死
- 不是 DataLoader 首 batch 获取失败
- 不是单卡前向本身失败
- 不是 `broad_cast_batch()` 的单卡路径问题
- 不是 `get_input()` / `encode_first_stage()` / `loss forward` 的单卡基础链路问题
- 不是 bitsandbytes warning 的直接致命报错

### 2.2 已确认的风险点
- 当前 SAT checkpoint loader 的 `load` 语义是：传父目录，读取 `latest`，再拼 iteration 子目录
- 若直接把 `load` 写成某个具体 iteration 目录（如 `.../1000`），会被拼成 `.../1000/1000/...`
- 因此“从 1000 恢复”需要通过受控方式实现，不能直接改 yaml 为 iteration 子目录

### 2.3 当前最可疑位置
训练问题大概率位于多卡训练第一步的后半段：
- `forward_step` 之后
- `all_reduce(loss / metrics)`
- `backward_step()`
- `model.step()` / DeepSpeed optimizer step

## 3. 目标

### 主目标
修复 Stage 3 训练，使其能够越过 `iteration 0` 并稳定输出第一条训练日志。

### 次目标
- 修复或规避 checkpoint 恢复入口的路径歧义
- 保留当前 `soft_focal + router schedule` 方案，不因调试误伤新的 loss 逻辑
- 给后续正式训练留下可复用的诊断与验证路径

## 4. 方案选择

评估过三条路径：

1. 直接继续重启正式训练
2. 先退回单卡绕过问题
3. 先做单步训练诊断，定位第一步训练挂点，再做最小修复

最终选择 **方案 3**，原因：
- 当前阻塞点显然不是“前向是否能跑”，而是“多卡训练第一步是否能完整结束”
- 直接反复重启只会继续浪费卡时
- 单卡绕过虽然能先跑，但不能解决多卡真实问题

## 5. 设计

### 5.1 诊断层
使用独立诊断脚本，不直接污染主训练入口。诊断脚本只负责定位：
- before / after `forward_step`
- before / after `all_reduce`
- before / after `backward`
- before / after `step`

这样可以精确定位第一步训练卡在哪个边界。

### 5.2 修复层
基于诊断结果做最小修复：
- 若是 checkpoint 恢复入口问题：修恢复逻辑或提供受控入口
- 若是 distributed reduce/backward/optimizer 问题：只修对应训练路径，不动无关模块
- 不做“顺手重构”

### 5.3 验证层
修复后按三层验证：
1. **单步训练验证**：完整走通 forward → reduce → backward → step
2. **短程训练验证**：正式短程训练越过 `iteration 0`
3. **正式训练验证**：目标配置下训练持续输出 step / loss 日志

## 6. 具体实施策略

### 阶段 A：定位挂点
- 在授权的其他空闲节点上运行 3-rank / 4-rank 单步训练诊断
- 确认卡点是在：
  - `forward_step`
  - `all_reduce`
  - `backward`
  - `model.step()`

### 阶段 B：修复恢复入口
- 不再直接把 yaml `load` 改到某个 iteration 子目录
- 需要时提供一个显式恢复入口，用于受控地从 1000 checkpoint 恢复

### 阶段 C：最小修复训练路径
- 只改根因所在的训练边界
- 若需额外日志，仅保留对排障有价值的边界日志
- 不引入和任务无关的结构性改造

### 阶段 D：重新启动训练
- 先跑短程验证
- 再启动正式训练
- 用日志确认持续推进而非再次停在 `iteration 0`

## 7. 成功标准

修复完成需满足：
- 训练不再停在 `iteration 0`
- 能输出至少第一条 iteration / loss 日志
- 不再出现错误的 checkpoint 路径拼接
- 至少有一次短程训练成功越过首步，并能作为正式训练前验证

## 8. 非目标

以下内容不在本次修复范围内：
- 重新设计 SF loss 体系
- 重构整个训练框架
- 优化 bitsandbytes / xformers 环境
- 修改论文实验逻辑本身

## 9. 风险与约束

- gpu2 是共享节点，继续正式训练前必须遵守占用检查规范
- 临时允许使用其他节点仅用于诊断，不应影响他人进程
- 训练恢复逻辑和 SAT 框架的 checkpoint 语义耦合较深，修复时必须保持兼容性

## 10. 下一步

1. 完成单步训练诊断，锁定挂点
2. 修复 checkpoint 恢复入口或训练边界问题
3. 验证单步训练通过
4. 验证短程训练越过 `iteration 0`
5. 重启目标训练配置
