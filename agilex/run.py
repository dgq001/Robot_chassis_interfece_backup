from pydemo import HttpClient, WSClient
ws_url = "ws://192.168.1.102:9090"  # 填写实际的机器人IP地址
http_url = "http://192.168.1.102:8880"
token = None
import time

if __name__ == '__main__':
    ### Http 客户端
    http_client = HttpClient(http_url)
    http_client.login_()


    # http_client.get_maplist()
    # print(http_client.map_list)
    # m = http_client.get_map_info("001")
    # print(m)
    # http_client.get_map_png("001")
    ### websocket 客户端
    ws_client = WSClient(ws_url)
    # ws_client.light_control()
    # for _ in range(20):
    #     ws_client.right_turn()
    #     time.sleep(0.1)

    # ws_client.move()
    # for _ in range(52):
    #     ws_client.move()
    #     time.sleep(0.1)
    """
    建图
    """

    print(ws_client.record_bag("start", "006"))

    stop_flag = input('输入任意字符结束录制:')

    http_client = HttpClient(http_url)
    http_client.login_()
    ws_client = WSClient(ws_url)

    print(ws_client.record_bag("stop", "006"))
    time.sleep(5)

    print(ws_client.mapping_3d("start", "006"))
    while True:
        try:
            print(ws_client.map_progress())
            time.sleep(0.5)
        except ConnectionResetError:
            http_client = HttpClient(http_url)
            http_client.login_()
            ws_client = WSClient(ws_url)
            break
    print(ws_client.mapping_3d("stop", "006"))
    time.sleep(5)
    #
    print(ws_client.mapping_2d("start", "006"))

    while True:
        try:
            print(ws_client.map_progress())
            time.sleep(0.5)
        except ConnectionResetError:
            http_client = HttpClient(http_url)
            http_client.login_()
            ws_client = WSClient(ws_url)
            break
    print(ws_client.mapping_2d("stop", "006"))

    """
    导航
    """
    # 终止上一次导航
    # ws_client.follow_line(idtype="stop")
    # time.sleep(2)
    ### 启动导航
    ws_client.follow_line(idtype="start", filename="006")

    time.sleep(2)

    # http_client.run_realtime_task([0.6, -0.2, 90])
    http_client.run_realtime_task([1, 1.0, 90])
    #关闭导航
    time.sleep(10)
    ws_client.follow_line(idtype="stop")
    #
    ws_client.on_close()
