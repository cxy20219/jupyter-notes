def progressbar(now:int,end:int,starttime:float,moth = 0):
    """
        now : 现在的次数
        end : 尾数
        starttime : 开始时间
        moth: int
        moth 0 : 不换行 
        moth 1 : 换行
    """
    sum =  end - 1
    num = now / sum / 0.01
    if moth == 0:
        if now == end - 1:
            print("\r当前进度 {:.2%}: ".format(num/100), "▇" * int(num)+"({:.1f}s)".format(time.time()-starttime) , flush = True)
        else:
            print("\r当前进度 {:.2%}: ".format(num/100), "▇" * int(num)+"({:.1f}s)".format(time.time()-starttime) , end="" , flush = True)
    else:
        print("当前进度 {:.2%}: ".format(num/100), "▇" * int(num)+"({:.1f}s)".format(time.time()-starttime), flush = True)