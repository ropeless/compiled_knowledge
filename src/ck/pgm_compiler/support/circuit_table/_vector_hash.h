#ifndef _VECTOR_HASH_H
#define _VECTOR_HASH_H

#include <vector>

// Inject hash function for std::vector<int> into namespace std.
namespace std
{
    template<>
    struct hash<std::vector<int>>
    {
        inline size_t operator()(std::vector<int> key) const noexcept
        {
            size_t h = 0;
            for (size_t i : key)
            {
                h = (h * 31) + i;
            }
            return h;
        }
    };
}


// For debugging...
namespace ck
{
    inline size_t hash_std_vector_int(std::vector<int> key)
    {
        std::hash<std::vector<int>> hasher;
        return hasher(key);
    }
}

#endif // _VECTOR_HASH_H
