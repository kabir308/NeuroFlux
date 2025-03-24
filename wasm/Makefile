# Compilateur WebAssembly
WASMC = wasm-ld

# Dossier des modèles
MODELS_DIR = ../models

# Dossier de sortie
OUT_DIR = .

# Modèles à compiler
MODELS = tinybert.onnx mobilenet.onnx

# Règle générale
all: $(addprefix $(OUT_DIR)/, $(MODELS:.onnx=.wasm))

# Règle de compilation
$(OUT_DIR)/%.wasm: $(MODELS_DIR)/%.onnx
	$(WASMC) -o $@ $<

clean:
	rm -f $(OUT_DIR)/*.wasm

.PHONY: all clean
