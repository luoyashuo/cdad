# 任务
目前的训练逻辑：scripts/grpo/start.sh
目前的实现很是单独在verl项目外面实现了src，但是我希望你把我自定义的reward计算逻辑直接集成到verl项目里面
# 要求
1. 你不可以修改数据配置，只能用目前提供好的数据格式
2. 所有模型都必须运行在GPU上
3. scripts/grpo/start.sh的运行前后的占卡逻辑不能够修改，但是你修改后仍然用scripts/grpo/start.sh来启动训练