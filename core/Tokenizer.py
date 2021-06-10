class Tokenizer:
    def __init__(self, token_path, start_token="<SOS>", end_token="<EOS>", pad_token="<PAD>"):
        self.tokens = [start_token, end_token, pad_token]
        for path in token_path:
            with open(path, "r") as fd:
                all_tokens = fd.read()
                for token in all_tokens.split("\n"):
                    if token not in self.tokens:
                        self.tokens.append(token)

        self.token_to_id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id_to_token = {i: tok for i, tok in enumerate(self.tokens)}

        self.START_TOKEN = start_token
        self.END_TOKEN = end_token
        self.PAD_TOKEN = pad_token
        self.SPECIAL_TOKENS = [self.START_TOKEN, self.END_TOKEN, self.PAD_TOKEN]

    def encode(self, sent):
        encode_tokens = []
        sent_tokens = sent.split()
        for token in sent_tokens:
            if token not in self.token_to_id:
                raise Exception("Truth contains unkown token")
            encode_tokens.append(self.token_to_id[token])

        if "" in encode_tokens:
            encode_tokens.remove("")

        return encode_tokens

    def decode(self, tokens, do_eval=False):
        decode_sent = ""
        for token in tokens:
            token = token.item()
            # decode excluding SPECIAL TOKENS.
            if do_eval:
                if self.id_to_token[token] not in self.SPECIAL_TOKENS:
                    decode_sent += self.id_to_token[token] + " "

            # decode all tokens.
            else:
                decode_sent += self.id_to_token[token] + " "

            # if encount END token, stop decoding.
            if self.id_to_token[token] == self.END_TOKEN:
                break

        return decode_sent
