# 代码运行逻辑
> **不要使用exit退出容器**
## 修改代码
1. 在本地文件修改
2. cd 到服务器的/mnt/d/data/workspace/guan/ada-mixer下面
3. git pull拉取最新代码

## 运行代码
1. 在服务器中
2. 使用docker exec -it ada /bin/bash命令进入容器，然后cd到/ada-mixer下面
3. 输入运行指令

## 提示
1. 建议开两个xshell标签页，一个放docker中的环境(用于运行代码)，一个放ada-mixer路径下(用于同步文件)
2. 退出容器时使用ctrl+P Q(ctrl不送，然后按P再按Q)
3. 如果上次使用ctrl +P Q的方式退出容器，则可以使用docker attach ada进入容器，并且会是上次退出时的命令行
4. **千万不要使用exit退出容器！！！！！**,用exit退出容器会使容器停止，代码运行也会终止
5. 容器内的/ada-mixer的代码与服务器的/mnt/d/data/workspace/guan/ada-mixer的代码是完全同步的，不要进行任何修改操作，代码的修改在本地的pycharm中进行。