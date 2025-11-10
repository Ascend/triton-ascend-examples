<div align="center">
<p align="center">
   <h1>Triton NPU编程案例</h1>
</p>
</div>

## 简介
GPU和NPU存在体系结构的一些差异，基于GPU开发的Triton算子无法在NPU上直接得到最佳性能，Ascend Triton样例仓提供了一些代码样例供开发者学习参考，旨在帮助开发者快速掌握在昇腾上编写Triton算子的一些优化写法，以及基于GPU优化的Triton算子如何快速迁移到NPU实现更好的性能。

## 试用范围
本案例仓的一些参考案例，适用于A2/A3等昇腾硬件。

## Tutorial目录结构

```
-basic：基础优化的案例，一些common case。
-op_extension: Ascend针对Triton扩展的一些Op语义，旨在利用昇腾硬件，实现最优性能
-best_practice：一些完整算子的优化案例，包含多个优化点，整体实现算子性能的最优
```

## 算子案例
#### 1、分核调整
基于GPU开发及优化的Triton算子，数据分片小，Kernel调用次数多，在迁移到NPU上运行或者开发者基于NPU原生开发Triton算子时，需要减少kernel调用次数，增加单kernel内处理的数据量，并在Kernel内做Tiling，结合double buffer提升流水并行，实现性能最优。
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [TilingExample01](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/basic/006-tiling.zh.md)| 基于昇腾硬件的实际物理核数，划分grid，并在kernel内做tiling，达成优化性能 | 
| [TilingExample02](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/basic/003-ub_overflow.zh.md)| 直接迁移基于GPU实现的Triton算子时，可能存在UB溢出问题，通过tiling解决UB内存消耗 |

#### 2、数据类型优化
A2/A3硬件矢量运算单元不支持部分特定数据类型，计算时会退化为标量运算，影响性能，在确定不影响精度的情况下，建议使用支持的数据类型，提升性能
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [vector cmp](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/basic/002-vector_cmp.zh.md) | Cmp Op不支持int32/int64数据类型矢量运算，不影响精度的情况下，转换为FP32，实现最优性能 |
| [vector add](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/basic/001-vector_add.zh.md) | 矢量运算单元不支持int64数据类型，不影响精度的情况下，使用int32，提升性能 | 

#### 3、离散访存
昇腾硬件上，Triton算子在批量读取数据时，如果数据存放位置不连续，会导致数据无法批量加载，退化为标量加载，大幅降低性能，下面总结了一些针对离散访存的列子，
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [discrete-memory-access01](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/basic/004-discrete_memory_access.zh.md) | 小数据量的离散访存，全部读取到UB后使用Gather语义筛选，替代从GM直接单个读取 |
| [discrete-memory-access02](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/002-decode_grouped_attention.zh.md) | 需要加载的Tensor在高维连续，低维离散时，转职实现向量化加载 |


#### 4、提升并行
提升指令并行度，有助于大幅提升性能
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [load order](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/basic/005-load_order.zh.md) | 将与计算没有依赖的数据加载语句提前，让MTE2指令并行下发，加速处理 |

#### 5、扩展语义
针对体系结构和访存的一些差异，昇腾提供一些Triton扩展语义，加速数据处理
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [InsertSlice](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/op_extension/001-insert_slice.zh.md) | 多个数据合并到一起，批量处理，提升数据处理效率 |
| [ExtractSlice](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/op_extension/002-extract_slice.zh.md) | 一次批量读取，截取部分数据处理，提升数据处理效率 |
| [LoadPadding](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/op_extension/003-load_care_padding.zh.md) | 带Mask的数据加载，被Mask掉部分如果不需要默认值，显示指定，提升MTE2与Vector的并行 |

#### 6、完整算子优化样列
基于具体开源模型接入的Triton算子，完整优化样列
|  **样例名称**  |  **样例介绍**  |
|---|---|
| [causal_conv1d](https://github.com/Ascend/triton-ascend-ops/blob/main/tutorial/best_practice/003-fused-cat-slice-conv1d.zh.md) | 基于SGLang QWen3-Next模型中causal_conv1d_update算子完整优化样列 |
