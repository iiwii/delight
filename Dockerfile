FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime AS base
RUN apt-get -y update && apt-get -y install wget unzip

FROM base as build
WORKDIR /fairseq
COPY . .
RUN pip install -e .

FROM build as wikitext-103
WORKDIR /workspace
COPY . .
RUN cd examples/language_model/ && bash prepare-wikitext-103.sh && cd ../..
RUN fairseq-preprocess \
        --only-source \
        --trainpref examples/language_model/wikitext-103/wiki.train.tokens \
        --validpref examples/language_model/wikitext-103/wiki.valid.tokens \
        --testpref examples/language_model/wikitext-103/wiki.test.tokens \
        --destdir data-bin/wikitext-103 \
        --workers 8
ENTRYPOINT [ "python", "lm_wikitext_103.py" ]
CMD [ "--d-m", "256" ]
