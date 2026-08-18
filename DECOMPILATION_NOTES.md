# Decompiled APK assessment

The supplied material is not the original source tree and is not sufficient for a safe Android rebuild.

## Evidence

- There is no Gradle wrapper, build script, dependency declaration, Kotlin source tree, or native build configuration.
- `Eagle Images Stacker.apk` is a Play split base package. Its manifest requires ABI and density splits, while this APK contains no `lib/<abi>/` native libraries. OpenCV/RAW native code therefore cannot be recovered from this file alone.
- The Java files are JADX reconstructions from five DEX files. Kotlin/Compose state and control flow have already been lowered and partially mis-reconstructed.
- `StackingProcess.java` contains duplicate local declarations and impossible conditions from decompilation. Treating it as editable source would introduce failures unrelated to the requested work.
- The recovered `BlendingMode` enum contains only `Average`, confirming the missing stack choices.

## Confirmed edge bug

The recovered astro path warps each accepted image into a black-bordered canvas, adds that canvas to one global accumulator, then divides every pixel by the global accepted-frame count. Pixels near a drift edge have fewer real samples, but the absent samples are still included in the divisor as black. This produces the reported brightness/SNR falloff and apparent softness.

AstroStack Studio instead warps a validity image beside every light frame and divides each output pixel by its own accumulated validity weight. Its regression test explicitly covers this case.

## Best route back to Android

Obtain the original project from the app owner, including Gradle files, signing-independent build configuration, and native libraries/source. The modules in `astrostack/` then provide precise, tested behavior to port into Kotlin/OpenCV without relying on invalid decompiler output.
