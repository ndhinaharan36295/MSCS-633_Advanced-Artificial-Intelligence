import qrcode

def generate_qr_code(url: str, output_file: str = "qrcode.png") -> None:
    """
    Generates a QR code for the given URL and saves it as an image file.

    Parameters:
        url (str): The URL to encode in the QR code.
        output_file (str): The name of the output image file (default: qrcode.png).
    """
    try:
        # Configure QR code settings
        qr = qrcode.QRCode(
            version=1,  # Controls the size of the QR Code (1–40, higher means bigger)
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
            box_size=10,  # Size of each "box" in pixels
            border=4,  # Border thickness (minimum is 4)
        )

        # Add data (URL) to the QR code
        qr.add_data(url)
        qr.make(fit=True)

        # Generate QR code image
        img = qr.make_image(fill_color="black", back_color="white")

        # Save image to file
        img.save(output_file)
        print(f"QR Code successfully generated and saved as {output_file}")

    except Exception as e:
        print(f"Error generating QR code: {e}")


if __name__ == "__main__":
    # Ask user for a URL
    user_url = input("Enter the URL to generate a QR code: ").strip()

    if user_url:
        generate_qr_code(user_url)
    else:
        print("No URL entered. Exiting...")
