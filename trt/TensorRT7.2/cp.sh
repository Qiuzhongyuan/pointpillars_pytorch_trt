#!/bin/bash
cp libnvcaffeparser.so.7.1.3 /usr/lib/x86_64-linux-gnu/ &&
cp libnvinfer_plugin.so.7.1.3 /usr/lib/x86_64-linux-gnu/ &&
cp libnvonnxparser.so.7.2.1 /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.7.1.3 &&
cp libnvinfer_plugin_static.a /usr/lib/x86_64-linux-gnu/ &&
cp libnvcaffeparser_static.a /usr/lib/x86_64-linux-gnu/
