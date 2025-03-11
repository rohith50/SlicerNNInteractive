mkdir -p /tmp/sam2_images
chmod -R 777 /tmp/sam2_images
# docker run -p 5565:5565 -p 5566:5566 -v /tmp/sam2_images:/home/user/tmp/sam2_images --gpus=all samurai_server
docker run -p 1526:1526 -v /tmp/sam2_images:/home/user/tmp/sam2_images samurai_server
