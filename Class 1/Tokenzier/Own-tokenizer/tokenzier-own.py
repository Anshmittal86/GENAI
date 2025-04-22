import string

class CharTokenizer:
    def __init__(self):
        self.char_vocab = {}
        self.id_to_char = {}
        self.id_counter = 0
        

        # English characters: a-z, A-Z, 0-9, punctuation, space
        english_chars = list(string.ascii_letters + string.digits + string.punctuation + ' ')

        # Hindi characters: vowels, consonants, matras, punctuation
        hindi_chars = list(
            "अआइईउऊऋएऐओऔअंअः" +
            "कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह" +
            "ािीुूृेैोौंःँ्" +
            "।॥"
        )
    
        # Add English characters to vocab
        for ch in english_chars:
            if ch not in self.char_vocab:
                self.char_vocab[ch] = self.id_counter
                self.id_counter += 1

        # Add Hindi characters to vocab
        for ch in hindi_chars:
            if ch not in self.char_vocab:
                self.char_vocab[ch] = self.id_counter
                self.id_counter += 1

        # Create reverse vocab
        for char, idx in self.char_vocab.items():
            self.id_to_char[idx] = char
        

    def encode(self, text):
        
        tokenIdList = []
        
        #Converting text to TokenIds and appending into list
        for ch in text:
            tokenId = self.char_vocab.get(ch, -1)
            tokenIdList.append(tokenId)
            
        return tokenIdList
        

    def decode(self, token_ids):
        
        text = ""
        
        # Converting tokenIds into text
        for token_id in token_ids:
            char = self.id_to_char.get(token_id, '')
            text += char
        
        return text


# ✅ Create tokenizer
tokenizer = CharTokenizer()

# 🔍 Test
tokens = tokenizer.encode("Hello123, चाय पीओ!")
print("Token IDs:", tokens)

text = tokenizer.decode(tokens)
print("Text: ", text)


