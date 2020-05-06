# car_face
2020-04-10
从AB两方面尝试改进效果：

A后处理策略
1、无车角状态帧舍去 
2、帧内策略：
1）没变的：frontleft多1算5,多3算5、frontright多2算5,多4算5这些没变
2）减小前后侧head_too_small选项的阈值 
3）backleft/backright策略：看到两个，则补345 
4）frontright策略：特殊策略修正2号错误定位：当2号head_x_min距离top_x_max参考位置大于head_x_center/2时，直接将之修正为5号。(sp1)
5）front策略：有5补34策略修改成：有5在帧内不补，在汇总后再另行处理(X 还是帧内处理，要求单张35or45同时出现)

3、汇总后策略（目前front和back都使用取并策略）：
1）右侧即使看不到2,也不再去掉2。
2）若汇总后有5,则如果3或4确实有出现在任意一帧，再补34

B.模型置信度：
1、考虑到目前FP较少，若将CONF设置到0.66,则原测试及牺牲FP一倍（原3%现6%），可获得自测数据上10%的识别提高


update04-11:
4、back设置有效区域3/5&2/5
5、模型再训：前+少量标注12人的fp，后+数据量。
注意：车边别站人

update04-12:
6、昨天的front模型训练完毕选择epoch20,识别提高，降低了前侧bboxsize约束
7、同时调整front有效区域：两对角之内（center往后加2个top_X）才有效（因为发现有后车司机情况）。
8、right front看不到司机，只看到单4几乎不可能发生，但又可能因为司机后仰而导致location错误，故这种情况4修正为1.  (sp2)
9、帧内：对前侧相机来说只有单张图上看到35或45才算345。
10、帧间：最终汇总时没2号，则把后排修改为34两人。
11、模型再重新训，加入了未正确的12345,并用cover bbox做压制。

update 04-15:
12、策略：再次调整（7）：front有效区域：两对角之内（center往后超过0.1倍长度都不算）才有效（因为发现有后车司机情况）。
13、策略：再次调整（10）：帧间：最终汇总时没2号，且当后排为3、4、5时，则把后排减去5。
14、策略：head_too_small在front还是要打开，因为有看穿后面保安的。
15、检测模型：第三次重新训练，再加入少量实测错误。
16、定位模型：实测中认为之前用的模型有点过拟合了，减少了之前定位的训练次数。

update 04-17
17、策略：后排T型定位排除干扰策略加入。
18、检测模型：前侧检测模型更新、后侧新检测模型启用。
19、前后排angle top head置信度分别根据目前模型更新了设置。