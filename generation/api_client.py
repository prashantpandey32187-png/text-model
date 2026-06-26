import requests
import time
import json
import os
from .config import POLLO_API_KEY, POLLO_BASE_URL


class PolloAIClient:
    """Client for Pollo AI Google Veo 3.1 video generation API."""

    def __init__(self):
        self.api_key = POLLO_API_KEY
        self.base_url = POLLO_BASE_URL
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

    def _raise_on_error(self, response: requests.Response, context: str = ""):
        """Check response and raise RuntimeError with API error message if failed."""
        if not response.ok:
            try:
                error_data = response.json()
                error_msg = error_data.get("message", error_data.get("errors", response.text))
            except (json.JSONDecodeError, ValueError):
                error_msg = response.text
            raise RuntimeError(
                f"API request failed (HTTP {response.status_code}){f' [{context}]' if context else ''}: {error_msg}"
            )

    def generate_video(
        self,
        prompt: str,
        image_paths: list = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration: int = 8,
        seed: int = None,
        generate_audio: bool = True,
        webhook_url: str = None,
        client_source: str = None,
        poll_interval: int = 5,
        max_wait_time: int = 600,
    ) -> bytes:
        """
        Generate a video from a text prompt using Google Veo 3.1 via Pollo AI.

        Args:
            prompt: Text description of the video
            image_paths: Optional list of image file paths to use as reference
            aspect_ratio: Aspect ratio ('16:9', '9:16', '1:1')
            resolution: Resolution ('720p', '1080p')
            duration: Video duration in seconds (default: 8)
            seed: Optional random seed for reproducibility
            generate_audio: Whether to generate audio (default: True)
            webhook_url: Optional webhook URL for async callback
            client_source: Optional client source identifier
            poll_interval: Seconds between status polls (default: 5)
            max_wait_time: Maximum seconds to wait for completion (default: 600)

        Returns:
            Raw video file bytes
        """
        # Build the request payload
        payload = {
            "input": {
                "prompt": prompt,
                "aspectRatio": aspect_ratio,
                "resolution": resolution,
                "duration": duration,
                "generateAudio": generate_audio,
            }
        }

        # Add optional images
        if image_paths:
            # Convert image paths to data URIs or file references
            image_data = []
            for img_path in image_paths:
                if os.path.exists(img_path):
                    import base64
                    with open(img_path, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode("utf-8")
                    ext = os.path.splitext(img_path)[1].lower()
                    mime = "image/png" if ext == ".png" else "image/jpeg"
                    image_data.append(f"data:{mime};base64,{encoded}")
            if image_data:
                payload["input"]["images"] = image_data

        # Add optional seed
        if seed is not None:
            payload["input"]["seed"] = seed

        # Add optional webhook URL
        if webhook_url:
            payload["webhookUrl"] = webhook_url

        # Add optional client source
        if client_source:
            payload["clientSource"] = client_source

        print(f"   Starting video generation...")
        print(f"   Prompt: {prompt}")
        print(f"   Resolution: {resolution}, Duration: {duration}s")

        # Submit the generation request
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
        )
        self._raise_on_error(response, "generate_video")

        result = response.json()
        print(f"   [OK] Generation submitted. Response: {json.dumps(result, indent=2)[:500]}")

        # Check if the response contains a direct download URL or requires polling
        video_url = None

        # Try to extract video URL from response
        if isinstance(result, dict):
            # Check various possible response formats
            if "output" in result:
                output = result["output"]
                if isinstance(output, list) and len(output) > 0:
                    video_url = output[0] if isinstance(output[0], str) else output[0].get("video", output[0].get("url"))
                elif isinstance(output, str):
                    video_url = output
                elif isinstance(output, dict):
                    video_url = output.get("video", output.get("url"))
            elif "data" in result:
                data = result["data"]
                if isinstance(data, list) and len(data) > 0:
                    video_url = data[0] if isinstance(data[0], str) else data[0].get("video", data[0].get("url"))
                elif isinstance(data, str):
                    video_url = data
            elif "video" in result:
                video_url = result["video"]
            elif "url" in result:
                video_url = result["url"]

        if video_url:
            # Download the video
            print(f"   Downloading video from: {video_url}")
            video_response = requests.get(video_url, stream=True)
            video_response.raise_for_status()
            return video_response.content
        else:
            # If response contains an ID, we might need to poll for status
            generation_id = None
            if isinstance(result, dict):
                generation_id = result.get("id") or result.get("generationId") or result.get("requestId")

            if generation_id:
                print(f"   Generation ID: {generation_id}. Polling for completion...")
                return self._poll_for_result(generation_id, poll_interval, max_wait_time)
            else:
                # Return raw response as fallback
                print(f"   Returning raw response content ({len(response.content)} bytes)")
                return response.content

    def _poll_for_result(
        self,
        generation_id: str,
        poll_interval: int = 5,
        max_wait_time: int = 600,
    ) -> bytes:
        """
        Poll for generation result until completion or timeout.

        Args:
            generation_id: The generation request ID
            poll_interval: Seconds between polls
            max_wait_time: Maximum seconds to wait

        Returns:
            Raw video file bytes

        Raises:
            RuntimeError: If generation fails or times out
        """
        # Derive the status URL from the base URL
        status_base = self.base_url.rstrip("/")
        status_url = f"{status_base}/{generation_id}"

        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            print(f"   Polling status... ({int(time.time() - start_time)}s elapsed)")
            response = requests.get(status_url, headers=self.headers)
            self._raise_on_error(response, "poll_status")

            result = response.json()
            status = None

            if isinstance(result, dict):
                status = result.get("status", result.get("state", "")).lower()

            # Completed successfully
            if status in ("completed", "succeeded", "done", "finished"):
                print(f"   Generation completed!")
                video_url = None
                if isinstance(result, dict):
                    output = result.get("output", result.get("data", result.get("result", {})))
                    if isinstance(output, list) and len(output) > 0:
                        video_url = output[0] if isinstance(output[0], str) else output[0].get("video", output[0].get("url"))
                    elif isinstance(output, str):
                        video_url = output
                    elif isinstance(output, dict):
                        video_url = output.get("video", output.get("url"))

                if video_url:
                    print(f"   Downloading video from: {video_url}")
                    video_response = requests.get(video_url, stream=True)
                    video_response.raise_for_status()
                    return video_response.content
                else:
                    print(f"   Raw result: {json.dumps(result, indent=2)[:500]}")
                    return response.content

            # Failed
            elif status in ("failed", "error", "cancelled"):
                error_msg = result.get("error", result.get("message", "Unknown error"))
                raise RuntimeError(f"Video generation failed: {error_msg}")

            # Still processing
            else:
                print(f"   Status: {status or 'processing'}...")
                time.sleep(poll_interval)

        raise RuntimeError(f"Video generation timed out after {max_wait_time}s")

    def text_to_video(
        self,
        prompt: str,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        duration: int = 8,
        seed: int = None,
        generate_audio: bool = True,
    ) -> bytes:
        """
        Simplified text-to-video generation.

        Args:
            prompt: Text description of the video
            aspect_ratio: Aspect ratio
            resolution: Resolution
            duration: Duration in seconds
            seed: Optional seed
            generate_audio: Whether to include audio

        Returns:
            Raw video file bytes
        """
        print(f"\n{'='*60}")
        print(f"   Pollo AI Video Generation")
        print(f"{'='*60}")
        print(f"   Prompt:            {prompt}")
        print(f"   Aspect Ratio:      {aspect_ratio}")
        print(f"   Resolution:        {resolution}")
        print(f"   Duration:          {duration}s")
        print(f"   Generate Audio:    {generate_audio}")
        print(f"{'='*60}\n")

        return self.generate_video(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration=duration,
            seed=seed,
            generate_audio=generate_audio,
        )