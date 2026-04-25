PATH = "/Users/yatharthnehva/Downloads/202601011400.pcap"

# Safe PCAP Inspector for VS Code / Python
# ---------------------------------------
# Purpose:
# 1. Open a .pcap file safely without loading everything into RAM
# 2. Read only the FIRST packet
# 3. Automatically discover available fields
# 4. Print ONE sample row
#
# Install first:
# pip install scapy

from scapy.all import PcapReader

PCAP_FILE = PATH   


def packet_to_dict(pkt):
    """
    Recursively extract all available fields from all layers.
    No hardcoding.
    """
    data = {}

    layer = pkt
    layer_num = 1

    while layer:
        layer_name = layer.__class__.__name__

        try:
            fields = layer.fields
            for key, value in fields.items():
                col_name = f"{layer_name}.{key}"
                data[col_name] = value
        except:
            pass

        # move to next layer
        layer = layer.payload
        layer_num += 1

        # stop if no real payload left
        if layer.__class__.__name__ == "NoPayload":
            break

    # generic packet info
    data["packet_total_length"] = len(pkt)

    return data


def main():
    print("Opening file safely...\n")

    try:
        with PcapReader(PCAP_FILE) as pcap:
            pkt = next(pcap)   # only first packet
            row = packet_to_dict(pkt)

            print("=" * 80)
            print("FIELDS FOUND IN FIRST PACKET:")
            print("=" * 80)

            for i, key in enumerate(row.keys(), 1):
                print(f"{i:02d}. {key}")

            print("\n" + "=" * 80)
            print("ONE SAMPLE ROW:")
            print("=" * 80)

            for key, value in row.items():
                print(f"{key}: {value}")

    except StopIteration:
        print("No packets found in file.")
    except FileNotFoundError:
        print("PCAP file not found.")
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    main()