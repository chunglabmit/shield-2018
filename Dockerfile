FROM chunglabmit/nuggt
#
# Install terastitcher
#
RUN mkdir /shield-2018
RUN git clone https://github.com/abria/TeraStitcher /shield-2018/TeraStitcher
RUN mkdir /shield-2018/build-terastitcher
RUN cd /shield-2018/build-terastitcher;\
    cmake ../TeraStitcher/src;\
    make -j `nproc`;\
    make install    
#
# Install pystripe
#
RUN git clone https://github.com/chunglabmit/pystripe /shield-2018/pystripe
RUN cd /shield-2018/pystripe;pip3 install -r requirements.txt;pip3 install -e .
#
# Install TSV
#
RUN git clone https://github.com/chunglabmit/tsv /shield-2018/tsv
RUN cd /shield-2018/tsv;pip3 install -r requirements.txt;pip3 install -e .
#
# Install Phathom
#
RUN git clone https://github.com/chunglabmit/phathom /shield-2018/phathom
RUN cd /shield-2018/phathom;pip3 install -r requirements.txt;pip3 install -e .

