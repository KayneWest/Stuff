import PIL.Image
import sys
 
def stitching(image, shred_width, left_shred, right_shred):
    """Calculate the cost of stitching two shreds.
 
    Arguments:
    image -- grayscale image
    shred_width -- width of a shred
    left_shred -- index of the left shred
    right_shred -- index of the right shred
 
    The cost of shredding left_shred to right_shred is equal to the sum of the
    absolute differences between pixels in the last column of left_shred and
    the first column of right_shred.
 
    """
    w, h = image.size
    data = image.getdata()
 
    return sum([abs(data[y * w + (left_shred + 1) * shred_width - 1] -
                    data[y * w + right_shred * shred_width])
                for y in xrange(h)])
 
def unshred(image, shred_width):
    """Compute left-to-right order shreds must be placed to unshred the image.
 
    Arguments:
    image -- grayscale image
    shred_width -- width of a shred
 
    The return value is a list of ordered shred indexes that will unshred
    the image.
 
    """
    def unshred_inner(stitchings, shreds_left, unshred_order):
        """Inner function for a single unshredding step.
 
        Arguments:
        stitchings -- list of (left_shred, right_shred, stitching_cost) tuples
                      of all possible stitchings, sorted by stitching_cost
        shreds_left -- shreds left to unshred
        unshred_order -- current list containing the left-to-right order in
                         which shreds must be placed to get an unshredded image
 
        unshred_inner is a recursive function that builds unshred's return
        value. At each step, a new shred is stitched to its original position.
        The next shred to be stitched is chosen to be the one with the
        lowest stitching cost between a shred that might be stitched to
        the left of the current unshredded image i.e. prepended to the
        unshred_order list and a shred that might be stitched to the right
        of the current unshredded image i.e. appended to the undred_order
        list. In the first step, when unshred_order is empty, the first shred
        is chosen as the one with the lowest stitching cost in the stitchings
        list (which in this moment contains all possible stitchings between
        any two distinct shreds).
 
        """
        if shreds_left == 0:
            return unshred_order
        else:
            if not unshred_order:
                best = stitchings[0]
                unshred_order_next = [best[0], best[1]]
                shreds_left_next = shreds_left - 2
            else:
                best_prepend = filter(lambda x: x[1] == unshred_order[0],
                                      stitchings)
                best_append = filter(lambda x: x[0] == unshred_order[-1],
                                     stitchings)
 
                if best_prepend and best_append and \
                        best_prepend[0][2] < best_append[0][2] or best_prepend:
                    unshred_order_next = [best_prepend[0][0]] + unshred_order
                elif best_append:
                    unshred_order_next = unshred_order + [best_append[0][1]]
                else:
                    raise Exception("can't happen")
 
                shreds_left_next = shreds_left - 1
 
            # Remove from the stitchings list all stitchings that are not
            # possible anymore. Those are the ones which have
            # unshred_order_next[0] as the left shred and the ones that contain
            # unshred_order_next[-1] (the last element of unshred_order_next)
            # as the right shred.
            stitchings_next = filter(
                lambda x: x[0] != unshred_order_next[0] and \
                    x[1] != unshred_order_next[-1],
                stitchings)
 
            return unshred_inner(
                stitchings_next, shreds_left_next, unshred_order_next)
 
    shreds = image.size[0] / shred_width
    stitchings = sorted([(s, t, stitching(image, shred_width, s, t))
                         for s in xrange(shreds) for t in xrange(shreds)
                         if s != t],
                        cmp=lambda x, y: x[2] - y[2])
 
    return unshred_inner(stitchings, shreds, [])
 
 
if __name__ == "__main__":
    image = PIL.Image.open(sys.argv[1])
    shred_width = 32
    unshred_order = unshred(image.convert("L"), shred_width)
    unshredded = PIL.Image.new("RGB", image.size)
 
    for i, shred in enumerate(unshred_order):
        x1, y1 = shred_width * shred, 0
        x2, y2 = x1 + shred_width, image.size[1]
        unshredded.paste(image.crop((x1, y1, x2, y2)), (i * shred_width, 0))
 
    unshredded.save("unshredded.png", "PNG")
