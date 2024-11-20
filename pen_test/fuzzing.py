# git clone https://github.com/RuntimeVerification/Neuzz.git
# cd Neuzz
# pip install -r requirements.txt
# sudo apt-get install libc6:i386 libncurses5:i386 libstdc++6:i386
# sudo apt-get install gcc-multilib g++-multilib
# python3 neuzz.py --target ./path/to/your/target --input-dir ./inputs --output-dir ./output --timeout 5
# ./path/to/your/target ./output/crash_input
# readelf -a ./path/to/your/target
# gdb ./path/to/your/target
# (gdb) run ./output/crash_input
# (gdb) bt      # To see the backtrace of the crash
# (gdb) info locals  # To view local variables in the current function
