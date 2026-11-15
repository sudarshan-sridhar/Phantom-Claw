# Receives the phone's Larix stream (RTMP on port 1935) and hands it to OpenCV on udp://127.0.0.1:5000.
# Larix connection URL:  rtmp://<laptop address>:1935/live/cam
# Restarts by itself each time the phone disconnects.
while ($true) {
    ffmpeg -hide_banner -loglevel warning -listen 1 -i rtmp://0.0.0.0:1935/live/cam -an -c:v copy -f mpegts "udp://127.0.0.1:5000?pkt_size=1316"
    Start-Sleep -Seconds 1
}
