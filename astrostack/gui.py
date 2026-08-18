from __future__ import annotations

import queue
import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .io import SUPPORTED_EXTENSIONS, write_stack_outputs
from .models import ProcessingOptions, StackMode, StackRequest, StackResult
from .pipeline import stack_images


class AstroStackApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("AstroStack Studio")
        self.geometry("1050x760")
        self.minsize(900, 650)
        self.lights: list[Path] = []
        self.darks: list[Path] = []
        self.output = tk.StringVar(value=str((Path.cwd() / "output" / "astrostack_improved.tiff").resolve()))
        self.mask = tk.StringVar()
        self.mode = tk.StringVar(value=StackMode.SIGMA_CLIPPED.value)
        self.half_size = tk.BooleanVar(value=False)
        self.auto_crop = tk.BooleanVar(value=True)
        self.auto_brightness = tk.BooleanVar(value=True)
        self.foreground = tk.BooleanVar(value=False)
        self.light_pollution = tk.BooleanVar(value=False)
        self.hdr = tk.BooleanVar(value=False)
        self.stars = tk.BooleanVar(value=False)
        self.denoise = tk.BooleanVar(value=True)
        self.distortion = tk.BooleanVar(value=True)
        self.chromatic_aberration = tk.BooleanVar(value=True)
        self.coverage = tk.DoubleVar(value=98.0)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_result: StackResult | None = None
        self.last_output: Path | None = None
        self._build()
        self.after(100, self._poll_events)

    def _build(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Hint.TLabel", foreground="#666")
        root = ttk.Frame(self, padding=18)
        root.pack(fill="both", expand=True)
        ttk.Label(root, text="AstroStack Studio", style="Title.TLabel").pack(anchor="w")
        ttk.Label(root, text="Coverage-aware stacking for rotating skies and fixed foregrounds", style="Hint.TLabel").pack(anchor="w", pady=(0, 14))

        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True)
        controls = ttk.Frame(body, padding=(0, 0, 14, 0))
        preview_frame = ttk.Frame(body)
        body.add(controls, weight=2)
        body.add(preview_frame, weight=3)

        files = ttk.LabelFrame(controls, text="Frames", padding=10)
        files.pack(fill="x")
        self.lights_label = ttk.Label(files, text="No light frames selected")
        self.darks_label = ttk.Label(files, text="No dark frames selected")
        ttk.Button(files, text="Choose lights", command=self._choose_lights).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.lights_label.grid(row=0, column=1, sticky="w")
        ttk.Button(files, text="Choose darks", command=self._choose_darks).grid(row=1, column=0, sticky="ew", padx=(0, 8), pady=6)
        self.darks_label.grid(row=1, column=1, sticky="w")
        ttk.Button(files, text="Foreground mask", command=self._choose_mask).grid(row=2, column=0, sticky="ew", padx=(0, 8))
        ttk.Entry(files, textvariable=self.mask).grid(row=2, column=1, sticky="ew")
        files.columnconfigure(1, weight=1)

        settings = ttk.LabelFrame(controls, text="Stack and finish", padding=10)
        settings.pack(fill="x", pady=12)
        ttk.Label(settings, text="Stacking mode").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            settings,
            textvariable=self.mode,
            values=(StackMode.SIGMA_CLIPPED.value, StackMode.AVERAGE.value, StackMode.SUM.value),
            state="readonly",
            width=18,
        ).grid(row=0, column=1, sticky="e")
        checks = (
            ("Fast half-resolution preview", self.half_size),
            ("Auto coverage crop", self.auto_crop),
            ("Auto brightness", self.auto_brightness),
            ("Protect foreground (auto or mask)", self.foreground),
            ("Reduce light pollution", self.light_pollution),
            ("High dynamic range", self.hdr),
            ("Enhance star light", self.stars),
            ("Dynamic noise reduction", self.denoise),
            ("Correct alignment distortion", self.distortion),
            ("Correct chromatic aberration", self.chromatic_aberration),
        )
        for row, (label, variable) in enumerate(checks, 1):
            ttk.Checkbutton(settings, text=label, variable=variable).grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Label(settings, text="Minimum crop coverage").grid(row=11, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(settings, from_=20, to=100, increment=5, textvariable=self.coverage, width=7).grid(row=11, column=1, sticky="e", pady=(6, 0))
        settings.columnconfigure(1, weight=1)

        output_frame = ttk.LabelFrame(controls, text="Output", padding=10)
        output_frame.pack(fill="x")
        ttk.Entry(output_frame, textvariable=self.output).pack(side="left", fill="x", expand=True)
        ttk.Button(output_frame, text="Browse", command=self._choose_output).pack(side="left", padx=(8, 0))

        self.run_button = ttk.Button(controls, text="Stack images", command=self._start)
        self.run_button.pack(fill="x", pady=(12, 6))
        self.save_button = ttk.Button(controls, text="Save current result", command=self._save_result, state="disabled")
        self.save_button.pack(fill="x", pady=(0, 6))
        self.progress = ttk.Progressbar(controls, mode="determinate")
        self.progress.pack(fill="x")
        self.status = ttk.Label(controls, text="Ready", style="Hint.TLabel")
        self.status.pack(anchor="w", pady=(4, 0))

        self.preview = ttk.Label(preview_frame, text="The completed stack will appear here", anchor="center")
        self.preview.pack(fill="both", expand=True)
        log_frame = ttk.LabelFrame(preview_frame, text="Processing log", padding=6)
        log_frame.pack(fill="both", expand=False, pady=(10, 0))
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.pack(fill="both", expand=True)

    def _choose_lights(self) -> None:
        self.lights = self._choose_images("Choose light frames")
        self.lights_label.configure(text=f"{len(self.lights)} selected" if self.lights else "No light frames selected")

    def _choose_darks(self) -> None:
        self.darks = self._choose_images("Choose dark frames")
        self.darks_label.configure(text=f"{len(self.darks)} selected" if self.darks else "No dark frames selected")

    def _choose_images(self, title: str) -> list[Path]:
        patterns = " ".join(f"*{suffix}" for suffix in sorted(SUPPORTED_EXTENSIONS))
        return [Path(path) for path in filedialog.askopenfilenames(title=title, filetypes=(("Images", patterns), ("All files", "*.*")))]

    def _choose_mask(self) -> None:
        path = filedialog.askopenfilename(title="Choose white-foreground mask")
        if path:
            self.mask.set(path)
            self.foreground.set(True)

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save stacked image",
            defaultextension=".tiff",
            filetypes=(("16-bit TIFF", "*.tiff"), ("16-bit PNG", "*.png"), ("JPEG", "*.jpg")),
        )
        if path:
            self.output.set(path)

    def _start(self) -> None:
        if not self.lights:
            messagebox.showerror("No light frames", "Choose the light frames to stack first.")
            return
        options = ProcessingOptions(
            mode=StackMode(self.mode.get()),
            half_size=self.half_size.get(),
            min_coverage=self.coverage.get() / 100.0,
            auto_crop=self.auto_crop.get(),
            auto_brightness=self.auto_brightness.get(),
            protect_foreground=self.foreground.get(),
            foreground_mask=Path(self.mask.get()) if self.mask.get() else None,
            reduce_light_pollution=self.light_pollution.get(),
            hdr=self.hdr.get(),
            enhance_stars=self.stars.get(),
            dynamic_denoise=self.denoise.get(),
            correct_distortion=self.distortion.get(),
            correct_chromatic_aberration=self.chromatic_aberration.get(),
        )
        request = StackRequest(
            list(self.lights),
            list(self.darks),
            options,
            lambda stage, current, total: self.events.put(("progress", (stage, current, total))),
            lambda message: self.events.put(("log", message)),
        )
        output = self.output.get()
        self.run_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.last_result = None
        self.last_output = None
        self.progress.configure(value=0)
        self._append_log("Starting stack")
        threading.Thread(target=self._run, args=(request, output), daemon=True).start()

    def _run(self, request: StackRequest, output_path: str) -> None:
        try:
            result = stack_images(request)
            output = write_stack_outputs(output_path, result.image, result.coverage, result.foreground_mask)
            if not output.is_file():
                raise OSError(f"output was not created: {output}")
            self.events.put(("done", (output, result)))
        except Exception as error:
            self.events.put(("error", (error, traceback.format_exc())))

    def _save_result(self) -> None:
        if self.last_result is None:
            messagebox.showerror("Nothing to save", "Run a stack first.")
            return
        try:
            output = write_stack_outputs(
                self.output.get(),
                self.last_result.image,
                self.last_result.coverage,
                self.last_result.foreground_mask,
            )
            self.last_output = output
            self.output.set(str(output))
            self.status.configure(text=f"Saved {output}")
            messagebox.showinfo("Result saved", f"Saved to:\n{output}")
        except Exception as error:
            self._append_log(traceback.format_exc())
            messagebox.showerror("Save failed", str(error))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "progress":
                    stage, current, total = payload  # type: ignore[misc]
                    self.status.configure(text=stage)
                    self.progress.configure(value=100.0 * current / max(total, 1))
                elif kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    output, result = payload  # type: ignore[misc]
                    self.last_result = result
                    self.last_output = output
                    self.output.set(str(output))
                    self._show_preview(Path(output))
                    self.status.configure(text=f"Saved {output} | {len(result.accepted)} accepted, {len(result.rejected)} rejected")
                    self.run_button.configure(state="normal")
                    self.save_button.configure(state="normal")
                    messagebox.showinfo("Stack complete", f"Saved to:\n{output}")
                elif kind == "error":
                    error, details = payload  # type: ignore[misc]
                    self._append_log(details)
                    self.status.configure(text="Failed")
                    self.run_button.configure(state="normal")
                    self.save_button.configure(state="disabled")
                    messagebox.showerror("Stack failed", str(error))
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _append_log(self, message: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", f"{message}\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _show_preview(self, path: Path) -> None:
        image = Image.open(path)
        image.thumbnail((620, 430), Image.Resampling.LANCZOS)
        if image.mode not in {"RGB", "RGBA"}:
            image = image.convert("RGB")
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview.configure(image=self.preview_photo, text="")


def main() -> None:
    AstroStackApp().mainloop()


if __name__ == "__main__":
    main()
