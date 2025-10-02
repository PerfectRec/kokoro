#!/usr/bin/env python3
"""
Benchmark batching performance with larger text blocks (~300 characters each).
"""

import time
import scipy.io.wavfile as wavfile
from kokoro import KPipeline
import os

# Larger text samples (~300 characters each)
LARGE_TEXTS = [
    "Artificial intelligence has revolutionized the way we interact with technology. Machine learning algorithms can now recognize patterns in data that would be impossible for humans to detect manually. Deep neural networks have achieved remarkable success in tasks ranging from image recognition to natural language processing.",
    
    "The field of natural language processing has made tremendous progress in recent years. Modern language models can understand context, generate coherent text, and even translate between languages with remarkable accuracy. These advances have enabled new applications in virtual assistants, content generation, and automated translation services.",
    
    "Computer vision systems have become incredibly sophisticated, enabling machines to interpret and understand visual information from the world around them. From facial recognition to autonomous vehicles, these technologies are transforming industries and creating new possibilities for human-computer interaction and automation.",
    
    "Neural networks are inspired by the structure of the human brain, consisting of interconnected nodes that process information in parallel. Through training on large datasets, these networks learn to recognize complex patterns and make predictions. This approach has proven remarkably effective across a wide range of applications.",
    
    "The integration of artificial intelligence into everyday devices has become increasingly common. Smart speakers, recommendation systems, and personal assistants use AI to provide personalized experiences. As these technologies continue to evolve, they promise to make our interactions with technology more natural and intuitive.",
    
    "Data science combines statistical analysis, machine learning, and domain expertise to extract insights from complex datasets. Organizations across industries rely on data scientists to help them make informed decisions, optimize processes, and identify new opportunities. The demand for these skills continues to grow rapidly.",
    
    "Cloud computing has transformed how businesses operate by providing scalable, on-demand access to computing resources. Companies can now deploy applications globally without maintaining their own infrastructure. This flexibility has accelerated innovation and made powerful computing resources accessible to organizations of all sizes.",
    
    "Cybersecurity has become increasingly critical as our world becomes more connected. Protecting sensitive information from malicious actors requires sophisticated defense mechanisms, constant vigilance, and rapid response capabilities. AI and machine learning are playing an increasingly important role in detecting and preventing security threats.",
    
    "The Internet of Things connects billions of devices worldwide, generating vast amounts of data. Smart homes, wearable technology, and industrial sensors are just a few examples of how connected devices are changing our lives. This interconnectedness creates both opportunities and challenges for privacy and security.",
    
    "Quantum computing represents a paradigm shift in computational power, leveraging quantum mechanical phenomena to solve problems that are intractable for classical computers. While still in early stages, quantum computers show promise for applications in cryptography, drug discovery, and optimization problems that could transform multiple industries.",
]

def save_audio(audio, filename, sample_rate=24000):
    """Save audio tensor to WAV file."""
    wavfile.write(filename, sample_rate, audio.numpy())

def benchmark_sequential(pipeline, texts, voice, output_dir):
    """Benchmark sequential processing."""
    print(f"\n{'='*70}")
    print(f"SEQUENTIAL PROCESSING (Batch Size = 1)")
    print(f"{'='*70}")
    
    start_time = time.time()
    results = []
    
    for i, text in enumerate(texts):
        text_start = time.time()
        for result in pipeline(text=text, voice=voice):
            results.append(result)
        text_time = time.time() - text_start
        char_count = len(text)
        print(f"  Text {i+1:2d} ({char_count:3d} chars): {text_time:.3f}s - '{text[:60]}...'")
    
    total_time = time.time() - start_time
    
    # Save audio files
    os.makedirs(output_dir, exist_ok=True)
    for i, result in enumerate(results):
        filename = os.path.join(output_dir, f"sequential_{i:02d}.wav")
        save_audio(result.audio, filename)
    
    print(f"\n  Total time: {total_time:.3f}s")
    print(f"  Average per text: {total_time/len(texts):.3f}s")
    print(f"  Total characters: {sum(len(t) for t in texts)}")
    print(f"  Chars per second: {sum(len(t) for t in texts)/total_time:.1f}")
    print(f"  Saved {len(results)} audio files to {output_dir}/")
    
    return total_time, results

def benchmark_batched(pipeline, texts, voice, batch_size, output_dir):
    """Benchmark batched processing."""
    print(f"\n{'='*70}")
    print(f"BATCHED PROCESSING (Batch Size = {batch_size})")
    print(f"{'='*70}")
    
    start_time = time.time()
    all_results = []
    
    # Process in batches
    for batch_idx in range(0, len(texts), batch_size):
        batch_texts = texts[batch_idx:batch_idx + batch_size]
        batch_start = time.time()
        
        results = pipeline.generate_batch(
            texts=batch_texts,
            voice=voice,
            speed=1.0
        )
        
        batch_time = time.time() - batch_start
        all_results.extend(results)
        
        print(f"  Batch {batch_idx//batch_size + 1} ({len(batch_texts)} texts): {batch_time:.3f}s")
        for i, text in enumerate(batch_texts):
            print(f"    {i+1}. ({len(text):3d} chars) '{text[:50]}...'")
    
    total_time = time.time() - start_time
    
    # Save audio files
    os.makedirs(output_dir, exist_ok=True)
    for i, result in enumerate(all_results):
        filename = os.path.join(output_dir, f"batch{batch_size}_{i:02d}.wav")
        save_audio(result.audio, filename)
    
    print(f"\n  Total time: {total_time:.3f}s")
    print(f"  Average per text: {total_time/len(texts):.3f}s")
    print(f"  Total characters: {sum(len(t) for t in texts)}")
    print(f"  Chars per second: {sum(len(t) for t in texts)/total_time:.1f}")
    print(f"  Saved {len(all_results)} audio files to {output_dir}/")
    
    return total_time, all_results

def main():
    print("="*70)
    print("KOKORO TTS BATCHING BENCHMARK - LARGE TEXTS")
    print("="*70)
    
    # Setup
    print("\nInitializing pipeline...")
    pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
    voice = 'af_bella'
    output_base = 'batching_output'
    
    print(f"Device: {pipeline.model.device}")
    print(f"Voice: {voice}")
    print(f"Number of texts: {len(LARGE_TEXTS)}")
    
    # Show text statistics
    char_counts = [len(t) for t in LARGE_TEXTS]
    print(f"\nText statistics:")
    print(f"  Min characters: {min(char_counts)}")
    print(f"  Max characters: {max(char_counts)}")
    print(f"  Average characters: {sum(char_counts)/len(char_counts):.1f}")
    print(f"  Total characters: {sum(char_counts)}")
    
    print(f"\nSample texts:")
    for i, text in enumerate(LARGE_TEXTS[:3], 1):
        print(f"  {i}. ({len(text)} chars) {text[:80]}...")
    
    # Benchmark different batch sizes
    results = {}
    
    # Sequential (batch size 1)
    seq_time, seq_results = benchmark_sequential(
        pipeline, LARGE_TEXTS, voice, 
        os.path.join(output_base, 'sequential')
    )
    results['sequential'] = seq_time
    
    # Batch size 5
    batch5_time, batch5_results = benchmark_batched(
        pipeline, LARGE_TEXTS, voice, 5,
        os.path.join(output_base, 'batch_5')
    )
    results['batch_5'] = batch5_time
    
    # Batch size 10
    batch10_time, batch10_results = benchmark_batched(
        pipeline, LARGE_TEXTS, voice, 10,
        os.path.join(output_base, 'batch_10')
    )
    results['batch_10'] = batch10_time
    
    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY - LARGE TEXTS")
    print("="*70)
    
    total_chars = sum(len(t) for t in LARGE_TEXTS)
    
    print(f"\nProcessing {len(LARGE_TEXTS)} texts ({total_chars} total characters):")
    print(f"  Sequential (batch=1):  {results['sequential']:.3f}s  (baseline)")
    print(f"  Batched (batch=5):     {results['batch_5']:.3f}s  ({results['sequential']/results['batch_5']:.2f}x speedup)")
    print(f"  Batched (batch=10):    {results['batch_10']:.3f}s  ({results['sequential']/results['batch_10']:.2f}x speedup)")
    
    print(f"\nAverage time per text:")
    print(f"  Sequential:  {results['sequential']/len(LARGE_TEXTS):.3f}s")
    print(f"  Batch=5:     {results['batch_5']/len(LARGE_TEXTS):.3f}s")
    print(f"  Batch=10:    {results['batch_10']/len(LARGE_TEXTS):.3f}s")
    
    print(f"\nCharacters per second:")
    print(f"  Sequential:  {total_chars/results['sequential']:.1f} chars/s")
    print(f"  Batch=5:     {total_chars/results['batch_5']:.1f} chars/s")
    print(f"  Batch=10:    {total_chars/results['batch_10']:.1f} chars/s")
    
    print(f"\nAudio files saved to: {output_base}/")
    print(f"  - {output_base}/sequential/")
    print(f"  - {output_base}/batch_5/")
    print(f"  - {output_base}/batch_10/")
    
    # Calculate total audio duration
    total_duration = sum(len(r.audio) for r in seq_results) / 24000
    print(f"\nTotal audio generated: {total_duration:.2f} seconds")
    print(f"Real-time factor (sequential): {total_duration/results['sequential']:.2f}x")
    print(f"Real-time factor (batch=10): {total_duration/results['batch_10']:.2f}x")
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()

