# CineBrain-SF v1 Claude Pack

这个压缩包用于把 **CineBrain-SF v1** 的实现任务交给 Claude，目标是在 **CineBrain / CineSync 源代码** 基础上完成可运行代码，而不是从零重写一套仓库。

## 包含文件
- `01_PROJECT_BRIEF.md`：项目目标、科学问题、与 CineSync 的差异
- `02_METHOD_SPEC.md`：正式方法规格，含模块、输入输出、训练阶段、损失
- `03_REPO_AUDIT_CHECKLIST.md`：Claude 接手代码前必须先完成的仓库审计清单
- `04_IMPLEMENTATION_TASKS.md`：分阶段编码任务拆解
- `05_EXPERIMENT_PLAN.md`：实验与消融设计
- `06_ACCEPTANCE_TESTS.md`：验收标准与最小可运行检查
- `07_PROMPT_FOR_CLAUDE.md`：可以直接复制给 Claude 的主提示词
- `08_CONFIG_TEMPLATE.yaml`：建议的配置草案
- `09_RESULTS_TABLE_TEMPLATES.md`：结果表模板

## 使用建议
1. 先把整个文件夹发给 Claude。
2. 明确要求 Claude **优先复用现有 CineBrain/CineSync 代码结构**，不要新造一套平行仓库。
3. 要求 Claude 第一步先执行 `03_REPO_AUDIT_CHECKLIST.md`，输出“仓库映射报告”后再开始编码。
4. 每完成一个阶段，都让 Claude 按 `06_ACCEPTANCE_TESTS.md` 自检。
5. 如果源码结构与文档假设不同，优先保留源码命名与目录风格，只调整接口与模块挂接方式。

## 重要说明
这些文档是基于论文与方案设计写的**实现规格包**，不是对你本地仓库结构的逐文件事实描述。
如果本地 CineBrain 源码目录与文档中的示例路径不同，以本地仓库实际结构为准，并由 Claude 在“仓库审计”阶段做映射。