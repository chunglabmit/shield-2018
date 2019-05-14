#
# Run as:
# docker build . --build-arg token=<github token>
#
# where <github token> is a token that can read the private repos (temporary)
#
FROM chunglabmit/nuggt:latest
ARG token
RUN mkdir /shield-2018
#
# Set up to use token authentication in Github
#
RUN git config --global url."https://${token}@github.com".insteadOf "https://github.com"
#
# Install Phathom
#
RUN git clone https://github.com/chunglabmit/phathom /shield-2018/phathom
RUN cd /shield-2018/phathom;pip3 install -r requirements.txt;pip3 install -e .
#
# Install shield-2018
#
ADD . /shield-2018/shield-2018
WORKDIR /shield-2018/shield-2018
RUN pip3 install -r requirements.txt
RUN pip3 install --editable .
#
# Download assets from website
#
RUN download-assets --destination /allen-mouse-brain-atlas
#RUN download-assets --tutorial --destination /tutorial
#
# Install terastitcher
#
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
# Install Pystripe
#
RUN git clone https://github.com/chunglabmit/pystripe.git
RUN cd /pystripe; python3 setup.py install
#
# Remove .gitconfig which has the token
#
RUN rm /root/.gitconfig
