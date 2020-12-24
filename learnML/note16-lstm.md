[toc]


# lstm结构


而 LSTM 不同。除了从上一个时间点的隐藏层得到的信息之外，还引入了一个memeory state。LSTM 和 RNN 的主要区别就是引入了 memory state 和三个门来控制状态的变化。

对于 RNN 来说,RNN 从上一个时间点得到的信息有一个，就是从上一个时间点的隐藏层得到的信息。 假设上一步的隐藏层为 h。那么它做的就是将当前时间点的输入 X 和 h 拼接起来，进行仿射变换和非线性变换之后输出，并将隐藏层的状态传递到下个时间点。

而 LSTM 来说，从上一个时间点得到的信息有两个，一个和 RNN 相同，是上个时间点的隐藏层的信息 h，还有一个是 memory state c。LSTM 将 X 和 c 拼接起来之后，进行仿射变换和非线性变换之后，会得到三个门控的状态和当前细胞的状态。其中当前细胞的状态对应 rnn 部分。

1. forget gate
这个是用来控制 memory state 的信息有多少保留。forget gate 是值是 X 和 h 拼接起来进行仿射变换和非线性变换之后得到的。形状和 c 相同，由于进行的仿射变换是 sigmoid 变换，因此输出值在 0 - 1 之间。用来表示输出的程度。它会和 c 进行逐元素相乘来控制 c 的输出。如果值为1，说明c全部输出。如果值为 0，如果值在 0-1 之间，说明会 c 中的值部分保留。

2. input gate
input gate 是用来控制 LSTM 的 rnn cell 部分的值有多少被保留的。input gate 的值的计算 和 forget gate 的类似。也是 X 和 h 拼接起来进行仿射变换之后和 sigmoid 变换之后得到的 。

3. output gate
output gate 是用来控制更新后的 memory state c 是如何输出的。它的计算方式和其它门控相同。最终会和 memory state 进行 tanh 变换后的值相称。

gru跟lstm有什么区别？
