\section*{1. 模型的输入与输出定义 (Model Input and Output)}

在 RLT 框架中，主要包含两个模型：教师模型 (Teacher, $\pi_\theta$) 和学生模型 (Student, $\pi_s$)。

\subsection*{1.1 教师模型 $\pi_\theta$ (Teacher Model)}
这是被强化学习 (RL) 优化的目标模型。
\begin{itemize}
    \item \textbf{输入 (Input):} 问题 ($q_i$) 加上对应的标准答案 ($s_i$)。
    \item \textbf{输出 (Output):} 一步步的推理过程/解释，被称为“思考标记” (think tokens, $t_{o_i}$)。
\end{itemize}

\subsection*{1.2 学生模型 $\pi_s$ (Student Model)}
这是作为奖励计算器 (Reward Model) 的辅助模型。它在两种不同的上下文中被调用：
\begin{itemize}
    \item \textbf{情景 A（用于计算 $r^{\text{SS}}$）:}
    \begin{itemize}
        \item \textbf{输入:} 问题 ($q_i$) + 教师生成的思考过程 ($t_{o_i}$)。
        \item \textbf{输出:} 生成标准答案 ($s_i$) 的预测对数概率。
    \end{itemize}
    \item \textbf{情景 B（用于计算 $r^{\text{KL}}$）:}
    \begin{itemize}
        \item \textbf{输入:} 仅有问题 ($q_i$)（不包含标准答案）。
        \item \textbf{输出:} 生成教师思考过程 ($t_{o_i}$) 中每个 token 的概率分布。
    \end{itemize}
\end{itemize}


\section*{2. 奖励函数定义 (Reward Definitions)}

针对教师模型生成的单条解释 $o_i$，最终的总奖励 $r^{\text{RLT}}_i$ 由两部分组成：
\begin{equation}
    r^{\text{RLT}}_i = r^{\text{SS}}(o_i, s_i, q_i) - \lambda r^{\text{KL}}(o_i, s_i, q_i)
\end{equation}
其中 $\lambda$ 为平衡系数（论文中设定为 3）。

\subsection*{2.1 学生理解奖励 (Student Success Reward, $r^{\text{SS}}$)}
该项用于衡量学生在阅读了教师的推理过程后，能否以高概率得出正确答案。公式融合了目标答案 token 对数概率的平均值 (avg) 和最小值 (min)：
\begin{equation}
    r^{\text{SS}}(o_i, s_i, q_i) = \text{avg} \left\{ \log \pi_s^{s_i} \right\} + \alpha \min \left\{ \log \pi_s^{s_i} \right\}
\end{equation}
其中的条件概率定义为：
\begin{equation}
    \pi_s^{s_i} = \pi_s(s_i \mid t_{o_i}, q_i)
\end{equation}
$\alpha$ 为缩放系数（论文中设定为 0.01）。

\subsection*{2.2 逻辑连贯性惩罚 (Logical Continuation KL Penalty, $r^{\text{KL}}$)}
该项用于计算教师的推理步骤在学生视角下的合理性。计算教师分布（已知答案）与学生分布（未知答案）之间的 KL 散度，融合了平均值 (avg) 和最大值 (max)：
\begin{equation}
    r^{\text{KL}}(o_i, s_i, q_i) = \text{avg} \left\{ D_{\text{KL}} \left( \pi_\theta^{t_{o_i}} \parallel \pi_s^{t_{o_i}} \right) \right\} + \alpha \max \left\{ D_{\text{KL}} \left( \pi_\theta^{t_{o_i}} \parallel \pi_s^{t_{o_i}} \right) \right\}
\end{equation}
其中分布的具体条件定义如下：
\begin{align}
    \pi_\theta^{t_{o_i}} &= \pi_\theta(t_{o_i} \mid s_i, q_i) \quad \text{(教师视角：包含答案)} \\
    \pi_s^{t_{o_i}} &= \pi_s(t_{o_i} \mid q_i) \quad \text{(学生视角：不包含答案)}
\end{align}