# 当前训练数据：无think、完整对话上限4096

原1981对，同步排除94题，两组各保留1887题。验证集原188题按字节保留。
使用本地Qwen3.5-0.8B tokenizer，enable_thinking=False。长度包含system、user、assistant及模板结束标记。
任一组长度大于4096则两组同时排除；恰好4096保留。不改正文、不截断、不重选候选、不调用API。
训练文件为normal_correct_sft.jsonl和shortest_correct_sft.jsonl；两组题目、顺序、提示完全一致。
训练配置应设max_length=4096，不能继承repair的max_assistant_tokens=2048限制。
全部保留样本已检查无正文think标记且生成前缀token对齐。训练时仍应只监督assistant并检查collator。
excluded_pairs.json列出排除题ID和两组长度；pair_lengths.jsonl含原全部配对测量。原v3数据不变。
本次仅生成训练数据，未启动训练、未验证云端显存。
