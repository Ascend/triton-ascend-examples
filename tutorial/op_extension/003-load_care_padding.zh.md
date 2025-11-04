# 003-load_care_padding性能优化

## 背景说明

`tl.load`接口中有一个参数`other`，如果不填写该参数，在应用`mask`后，如果出现尾块，这些尾块会被填`0`值。

> 这里`mask = idx < M`中，如果`M`为常量，则不会触发本文提到的问题。如果`M`是变量，可参考本文进行优化。

在昇腾NPU的实现中，会分2步操作：

1. 将尾块内存都填成`other`的值。默认填`0`值，如果用户指定，则填用户指定的值。
2. 加载真实需要的值

在用户不需要`other`值的情况下，这样的操作存在2个性能影响点：

1. 多余的数据搬运操作
2. 在对尾块赋值时，加载真实的数据会被阻塞

## 优化方案

为解决上面的问题，tritan-ascend为`tl.load`接口增加一个参数：`care_padding`，基类型为`boolean`，默认为`True`。

如果用户在使用场景中，不需要用到`other`的值，则可以通过将`care_padding`设置为`False`，实现性能提升。此时，尾块中的值会是随机数。

例：
```python
idx = tl.arange(0, N)
mask = idx < M
tl.load(ptr + idx, mask = mask, care_padding=False)
```

如果同时指定了`other`和`care_padding`，则`care_padding`的设置会被忽略，即，使用默认值`True`。

> 注意：如果在用户的业务场景中，需要依赖`other`的0值，那么使用`care_padding=True`会出现非预期的结果。这种情况下，不建议使用。