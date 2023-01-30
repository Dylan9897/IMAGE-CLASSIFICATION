
AlexNet = [
        ["Conv2D",3,64,(5,5),(1,1),(0,0)],
        ["Maxpool",(3,3),(2,2),(0,0)],
        ["Conv2D",64,64,(5,5),(1,1),(0,0)],
        ["Maxpool",(3,3),(2,2),(0,0)],
        ["Conn",1024,384],
        ["Conn",384,192],
        ["LastLinear",192,10]
        ]

vgg16 = [
        ["Conv2D",3,64,(3,3),(1,1),(1,1)],
        ["Maxpool",(2,2),(2,2),(0,0)],
        ["Conv2D",64,128,(3,3),(1,1),(1,1)],
        ["Maxpool",(2,2),(2,2),(0,0)],
        ["Conv2D",128,256,(3,3),(1,1),(1,1)],
        ["Conv2D",256,256,(3,3),(1,1),(1,1)],
        ["Maxpool",(2,2),(2,2),(0,0)],
        ["Conv2D",256,512,(3,3),(1,1),(1,1)],
        ["Conv2D",512,512,(3,3),(1,1),(1,1)],
        ["Maxpool",(2,2),(2,2),(0,0)],
        ["Conv2D",512,512,(3,3),(1,1),(1,1)],
        ["Conv2D",512,512,(3,3),(1,1),(1,1)],
        ["Maxpool",(2,2),(2,2),(0,0)],
        ["Conn",512,100],
        ["LastLinear",100,10]
        ]


