"""Quantization formats and their properties."""

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantFormat:
    name: str
    bits_per_weight: float
    quality_rating: str  # "Low", "Fair", "Good", "Very Good", "Excellent", "Lossless"
    description: str


# GGUF quantization formats (llama.cpp)
QUANT_FORMATS: dict[str, QuantFormat] = {
    "Q2_K": QuantFormat(
        "Q2_K", 2.63,
        "Low",
        "2-bit quantization (k-quant). Significant quality loss, only for very large models.",
    ),
    "Q3_K_S": QuantFormat(
        "Q3_K_S", 3.41,
        "Low",
        "3-bit quantization small. Noticeable quality loss.",
    ),
    "Q3_K_M": QuantFormat(
        "Q3_K_M", 3.07,
        "Fair",
        "3-bit quantization medium. Better than Q3_K_S with minor size increase.",
    ),
    "Q3_K_L": QuantFormat(
        "Q3_K_L", 3.35,
        "Fair",
        "3-bit quantization large. Best 3-bit option.",
    ),
    "Q4_0": QuantFormat(
        "Q4_0", 4.50,
        "Fair",
        "4-bit quantization, legacy format. Use Q4_K_M instead.",
    ),
    "Q4_K_S": QuantFormat(
        "Q4_K_S", 4.58,
        "Good",
        "4-bit quantization small (k-quant). Good balance for smaller devices.",
    ),
    "Q4_K_M": QuantFormat(
        "Q4_K_M", 4.85,
        "Good",
        "4-bit quantization medium (k-quant). Recommended default for most users.",
    ),
    "Q5_0": QuantFormat(
        "Q5_0", 5.50,
        "Good",
        "5-bit quantization, legacy format. Use Q5_K_M instead.",
    ),
    "Q5_K_S": QuantFormat(
        "Q5_K_S", 5.54,
        "Very Good",
        "5-bit quantization small (k-quant). High quality, moderate size.",
    ),
    "Q5_K_M": QuantFormat(
        "Q5_K_M", 5.69,
        "Very Good",
        "5-bit quantization medium (k-quant). Near-original quality.",
    ),
    "Q6_K": QuantFormat(
        "Q6_K", 6.56,
        "Excellent",
        "6-bit quantization (k-quant). Very close to original quality.",
    ),
    "Q8_0": QuantFormat(
        "Q8_0", 8.50,
        "Excellent",
        "8-bit quantization. Virtually no quality loss.",
    ),
    "FP16": QuantFormat(
        "FP16", 16.0,
        "Lossless",
        "Full 16-bit floating point. Original model quality, largest size.",
    ),
}

# Recommended formats for different use cases
RECOMMENDED_FORMATS = ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "FP16"]
