[toc]

# Pytorch Transforms

## 常用变换

先读取图片，大 Egoist 镇楼！

```
from PIL import Image

img_path= "/Users/ed/Downloads/egoist.jpg"

img = Image.open(img_path)
img.show()
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331174755.png)

### Resize

```
from torchvision import transforms

resized = transforms.Resize((200, 400))(img)
resized.show()
print("{} ==> {}".format(img.size, resized.size)) # (1200, 630) ==> (400, 200)
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331175125.png)

### CenterCrop

```
centerCropped = transforms.CenterCrop(500)(img)
centerCropped.show()
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331175440.png)


### RandomCrop

```
randomCropped = transforms.RandomCrop(500)(img)
randomCropped.show()
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331175236.png)

### RandomResizedCrop

```
randomResizeCropped = transforms.RandomResizedCrop(200)(img)
randomResizeCropped.show()
```

可以看到不仅图片被crop 出一块500x500的大小，被裁剪部分的长宽比也变了（因为使用了先Resize后Crop，Resize会让图像的比例变化。

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331180944.png)

### RandomHorizontalFlip

```
h_fliped = transforms.RandomHorizontalFlip(p=1)(img)
h_fliped.show()
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331175539.png)

### RandomVerticalFlip

```
v_fliped = transforms.RandomVerticalFlip(p=1)(img)
v_fliped.show()
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331175641.png)


### RandomRotation

```
rotated = transforms.RandomRotation(45)(img)
rotated.show()
```
![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331175842.png)


### RandomGrayscale

```
grayed = transforms.RandomGrayscale(p=0.5)(img)
grayed.show()
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331175915.png)




### ColorJitter

```
brigited = transforms.ColorJitter(brightness=1)(img)
brigited.show()
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331180051.png)

```
constrasted= transforms.ColorJitter(constrast=1)(img)
constrasted.show()
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331180139.png)


```
saturated = transforms.ColorJitter(saturation=0.5)(img)
saturated.show()
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331180343.png)

```
hued = transforms.ColorJitter(hue=0.5)(img)
hued.show()
```

![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331180248.png)

###  Pad

```
padded = transforms.Pad((0, (img.size[0] - img.size[1]) // 2))(img)
padded.show()
```
![](https://gitee.com/EdwardElric_1683260718/picture_bed/raw/master/img/20200331180502.png)


## Compose

使用 Compose 将 transform 组合起来，之后会作为参数传入到 DataSet中

```
data_transform=transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

# References
- [图像分类：数据增强（Pytorch版） - 知乎](https://zhuanlan.zhihu.com/p/54527197)
